"""Test-split conditional likelihood with BF16 and a causal TRT wiring audit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import urllib.request

BASE = Path('/lustre/fsw/portfolios/nemotron/users/helenn')
FI = BASE/'flashinfer-nt4-gb300-nvls'
MC = BASE/'megatron-lm-nt4-coscheduling'


def fetch_corpus(dataset, config, tok, root):
    import pyarrow.parquet as pq
    url = 'https://datasets-server.huggingface.co/parquet?dataset=' + dataset
    with urllib.request.urlopen(url, timeout=120) as response:
        listing = json.load(response)
    files = [x for x in listing['parquet_files'] if x['config']==config and x['split']=='test']
    if not files:
        raise RuntimeError('No test parquet files for '+dataset)
    documents, seen, provenance = [], set(), []
    with tempfile.TemporaryDirectory(prefix='nt4-likelihood-data-') as directory:
        for index, item in enumerate(files):
            path = Path(directory)/f'{index}.parquet'
            with urllib.request.urlopen(item['url'], timeout=180) as response, path.open('wb') as output:
                while block := response.read(1024*1024):
                    output.write(block)
            provenance.append(dict(url=item['url'], sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for row_index, row in enumerate(pq.read_table(path).to_pylist()):
                text = row['text'] if 'text' in row else 'Question: '+row['question']+'\nAnswer: '+row['answer']
                if not text.strip() or text in seen:
                    continue
                seen.add(text)
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids)<192 or any(t in tok.all_special_ids for t in ids[:192]):
                    continue
                documents.append(dict(source_row=row_index,source_file=index,source_text=text,prefix_ids=ids[:64],target_ids=ids[64:192]))
                if len(documents)==128:
                    break
            if len(documents)==128:
                break
    if len(documents)!=128:
        raise RuntimeError(f'{dataset}: only {len(documents)} eligible passages')
    corpus = dict(dataset=dataset,config=config,split='test',provenance=provenance,vocab_size=len(tok),documents=documents,selection='First 128 distinct eligible >=192-token test rows; 64-token prefix, 128-token continuation. Length-biased. No certification against checkpoint training data. GSM8K includes ground-truth worked solutions, not model-generated continuations.')
    root.mkdir()
    (root/'corpus.json').write_text(json.dumps(corpus))
    (root/'prompts.jsonl').write_text(''.join(json.dumps(dict(text=tok.decode(d['prefix_ids'])))+'\n' for d in documents))


def run_arm(root, corpus_root, arm, source_command, audit=False):
    output=root/arm
    output.mkdir()
    command=list(source_command)
    command[0]=sys.executable
    command[5]=str(FI/'examples/moe_ep/nt4_teacher_forced_audit.py')
    if arm=='bf16':
        from nt4_e2e import precision_options
        command=precision_options(command,'bf16')
    options={'--prompt-file':corpus_root/'prompts.jsonl','--prompt-file-num-truncate':128,'--num-tokens-to-generate':128,'--termination-id':-1}
    if arm=='bf16':
        options['--inference-grouped-gemm-backend']='vllm'
    for key,value in options.items():
        if key in command:
            command[command.index(key)+1]=str(value)
        else:
            command.extend([key,str(value)])
    env={k:v for k,v in os.environ.items() if not k.startswith('NT4_')}
    env.update(PYTHONPATH=f'{FI}:{MC}',FLASHINFER_WORKSPACE_BASE=str(BASE),FLASHINFER_TRTLLM_NT4_EXPORT=str(BASE/'logs/nt4-trtllm-clamped-sweep-20261020-134327/export'),MAX_JOBS='4')
    for k,v in dict(ALLOW_SOURCE_JIT=1,DETACHED_TAIL=1,TE_STREAM_WORKSPACES=1,LOOKAHEAD_AFTER_MMA=0,PACKED_FC1_ACTIVATION=0,MXFP8_GROUP_HINT=1024,SCHED_LOOKUP=1,E2E_PRECISION='bf16' if arm=='bf16' else 'mxfp8',E2E_REQUEST_COUNT=128,E2E_ROUNDS=1,CONTROLLED_GC=1,SHORTCUT_OVERLAP=1,E2E_MODEL_GRAPHS=1,E2E_ROUTED_GRAPHS=0,HYBRID_MCORE_ROOT=str(MC),TANH_LOGIT_SOURCE=str(root),TEACHER_CORPUS=str(corpus_root/'corpus.json')).items():
        env['NT4_'+k]=str(v)
    hw,trt=arm=='megamoe-tanh',arm=='trtllm'
    mode='vllm' if arm=='bf16' else ('baseline' if arm=='cutlass' else 'direct')
    env.update(NT4_E2E_OUTPUT=str(output),NT4_E2E_MODE=mode,NT4_MXFP8_NVLS_HYBRID=str(int(mode=='direct')),NT4_TRTLLM_CLAMPED=str(int(trt)),NT4_TRT_EXPLORATORY_TIMING=str(int(trt)),NT4_VALIDATION_ISOLATE_TRT=str(int(trt)),NT4_LIKELIHOOD_TRT_AUDIT=str(int(trt and audit)))
    for k in ('HARDWARE_TANH','TANH_FC1_ADVISORY','TANH_EXPERT_ADVISORY','TANH_OUTPUT_ADVISORY'):
        env['NT4_'+k]=str(int(hw))
    if arm=='cutlass':
        env['NT4_CUTLASS_DECODE_PROFILE_IDS']='[39,428]'
    for k in ('CUDA_LAUNCH_BLOCKING','MEGA_USE_NCU'):
        env.pop(k,None)
    (output/'command.json').write_text(json.dumps(command,indent=2))
    (output/'experiment-env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('NT4_','FLASHINFER_'))},indent=2))
    print('Starting',output,flush=True)
    with (root/(arm+'.log')).open('w') as stream:
        p=subprocess.Popen(command,env=env,cwd=MC,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    (output/'exit.json').write_text(json.dumps(dict(returncode=code)))
    print('Finished',output,'exit',code,flush=True)
    return code


def summarize(root):
    import numpy as np
    arms=('cutlass','bf16','megamoe-exp','megamoe-tanh','trtllm')
    data={a:json.loads((root/a/'likelihood-round1.json').read_text()) for a in arms if (root/a/'likelihood-round1.json').exists() and json.loads((root/a/'exit.json').read_text())['returncode']==0}
    results={}
    for arm,d in data.items():
        values=np.array([[t['logprob'] for t in doc] for doc in d['documents']])[:,1:]
        row=dict(decode_tokens=int(values.size),nll=float(-values.mean()),perplexity=float(np.exp(-values.mean())),comparisons={})
        for ref,c in data.items():
            if (d['corpus_sha256'],d['schedule_sha256']) != (c['corpus_sha256'],c['schedule_sha256']):
                raise AssertionError('Inputs/schedules differ')
            control=np.array([[t['logprob'] for t in doc] for doc in c['documents']])[:,1:]
            delta=(control-values).mean(1)
            rng=np.random.default_rng(20261102)
            bootstrap=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
            row['comparisons'][ref]=dict(delta_nll=float(delta.mean()),document_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist())
        results[arm]=row
    (root/'summary.json').write_text(json.dumps(dict(results=results,missing=[a for a in arms if a not in data],scope='Conditional test-split likelihood, not downstream accuracy or certification of kernel correctness'),indent=2))


def main():
    from transformers import AutoTokenizer
    try:import nvshmem.core
    except ImportError:subprocess.run([sys.executable,'-m','pip','install','--no-deps','nvshmem4py-cu13==0.4.0'],check=True)
    root=BASE/('logs/nt4-likelihood-test-splits-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    root.mkdir();print('Results:',root,flush=True)
    command=json.loads((BASE/'logs/nt4-tanh-answer-screen-20261102-074414/batch0/hybrid-hardware-tanh-command.json').read_text())
    tok=AutoTokenizer.from_pretrained(command[command.index('--tokenizer-model')+1],local_files_only=True)
    # Audit on the old corpus first, so exact CUTLASS equality can be examined.
    old=BASE/'logs/nt4-teacher-likelihood-20261102-130452'
    audit=root/'trt-audit';audit.mkdir()
    if run_arm(audit,old,'trtllm',command,audit=True):
        print('TRT audit failed; TRT arm withheld, other backends will still run',flush=True)
        trt_ok=False
    else:trt_ok=True
    for name,dataset,config in [('wikitext','Salesforce/wikitext','wikitext-103-raw-v1'),('gsm8k','openai/gsm8k','main')]:
        group=root/name;group.mkdir()
        try:fetch_corpus(dataset,config,tok,group/'data')
        except Exception as error:
            (group/'data-error.json').write_text(json.dumps(dict(error=repr(error))))
            print('Dataset failed',name,repr(error),flush=True)
            continue
        for arm in ('cutlass','bf16','megamoe-exp','megamoe-tanh','trtllm'):
            if arm=='trtllm' and not trt_ok:continue
            run_arm(group,group/'data',arm,command)
        summarize(group)
    print('Queue complete:',root,flush=True)


if __name__=='__main__':main()

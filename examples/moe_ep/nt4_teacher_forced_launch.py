"""Run four fresh backend processes on identical OpenMath text continuations."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


def summarize(root):
    import numpy as np
    arms = ('cutlass', 'megamoe-exp', 'megamoe-tanh', 'trtllm')
    data = {arm: json.loads((root/arm/'likelihood-round1.json').read_text()) for arm in arms}
    control = data['cutlass']
    if any((d['corpus_sha256'], d['schedule_sha256']) != (control['corpus_sha256'], control['schedule_sha256']) for d in data.values()):
        raise AssertionError('Inputs or request schedules differ across arms')
    report = dict(scope='Conditional perplexity of math-problem continuations; not answer accuracy or certified held-out quality', performance_use=False, arms={})
    rng = np.random.default_rng(20261102)
    for arm, d in data.items():
        lp = np.array([[x['logprob'] for x in doc] for doc in d['documents']], dtype=np.float64)
        baseline = np.array([[x['logprob'] for x in doc] for doc in control['documents']], dtype=np.float64)
        # First target comes from prefill; remaining targets exercise decode substitutions.
        delta = baseline[:, 1:] - lp[:, 1:]
        doc_delta = delta.mean(axis=1)
        boots = doc_delta[rng.integers(len(doc_delta), size=(2000, len(doc_delta)))].mean(axis=1)
        top = np.array([[x['top_id'] for x in doc] for doc in d['documents']])
        ctrl_top = np.array([[x['top_id'] for x in doc] for doc in control['documents']])
        worst = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        report['arms'][arm] = dict(tokens=int(lp.size), mean_nll=float(-lp.mean()), conditional_perplexity=math.exp(float(-lp.mean())), decode_mean_nll=float(-lp[:, 1:].mean()), decode_conditional_perplexity=math.exp(float(-lp[:, 1:].mean())), delta_nll_vs_cutlass=float(delta.mean()), paired_document_bootstrap_95=np.quantile(boots,[.025,.975]).tolist(), decode_top1_agreement=float((top[:, 1:]==ctrl_top[:, 1:]).mean()), absolute_logprob_delta_quantiles=np.quantile(np.abs(delta),[.5,.9,.99,1]).tolist(), worst=dict(document=int(worst[0]),position=int(worst[1]+1),delta_nll=float(delta[worst])), numerical_qualification='Separate kernel checks apply; exploratory exceptions are NOT_PASSED')
    (root/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from transformers import AutoTokenizer
    base = Path('/lustre/fsw/portfolios/nemotron/users/helenn')
    fi = base/'flashinfer-nt4-gb300-nvls'
    mc = base/'megatron-lm-nt4-coscheduling'
    root = base/('logs/nt4-teacher-likelihood-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    root.mkdir()
    print('Results:', root, flush=True)
    source = base/'logs/nt4-tanh-answer-screen-20261102-074414/batch0'
    command = json.loads((source/'hybrid-hardware-tanh-command.json').read_text())
    tok_path = command[command.index('--tokenizer-model')+1]
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    shard = Path('/lustre/fsw/portfolios/nemotron/users/rkirby/hf_home/datasets/nvidia___open_math_instruct-2/default/0.0.0/469216e3f46f4dacf476b382e192485ea51a143e/open_math_instruct-2-train_5M-00000-of-00014.arrow')
    excluded = {json.loads(line)['text'].strip() for line in (base/'research/routing/openmath2_unique2048.jsonl').read_text().splitlines()}
    docs, seen = [], set()
    row_id = -1
    with pa.memory_map(str(shard), 'r') as f:
        for batch in ipc.open_stream(f):
            for problem in batch.column('problem').to_pylist():
                row_id += 1
                if not problem or problem.strip() in excluded or problem in seen:
                    continue
                seen.add(problem)
                ids = tok.encode(problem, add_special_tokens=False)
                if len(ids) < 192 or any(t in tok.all_special_ids for t in ids[:192]):
                    continue
                docs.append(dict(source_row=row_id, source_text=problem, prefix_ids=ids[:64], target_ids=ids[64:192]))
                if len(docs) == 128:
                    break
            if len(docs) == 128:
                break
    if len(docs) != 128:
        raise ValueError('Insufficient eligible documents')
    corpus = dict(source=str(shard), selection='First 128 distinct eligible rows, >=192 tokens, excluding tuning prompts and special tokens; length-biased training-shard diagnostic', tokenizer=tok_path, vocab_size=len(tok), documents=docs)
    corpus_path = root/'corpus.json'
    corpus_path.write_text(json.dumps(corpus))
    (root/'prompts.jsonl').write_text(''.join(json.dumps(dict(text=tok.decode(d['prefix_ids'])))+'\n' for d in docs))
    command[0] = sys.executable
    command[5] = str(fi/'examples/moe_ep/nt4_teacher_forced_eval.py')
    for key, value in {'--prompt-file':root/'prompts.jsonl','--prompt-file-num-truncate':128,'--num-tokens-to-generate':128,'--termination-id':-1}.items():
        if key in command:
            command[command.index(key)+1] = str(value)
        else:
            command.extend([key,str(value)])
    env = {k:v for k,v in os.environ.items() if not k.startswith('NT4_')}
    env.update(PYTHONPATH=f'{fi}:{mc}',FLASHINFER_WORKSPACE_BASE=str(base),FLASHINFER_TRTLLM_NT4_EXPORT=str(base/'logs/nt4-trtllm-clamped-sweep-20261020-134327/export'),MAX_JOBS='4')
    for k,v in dict(ALLOW_SOURCE_JIT=1,DETACHED_TAIL=1,TE_STREAM_WORKSPACES=1,LOOKAHEAD_AFTER_MMA=0,PACKED_FC1_ACTIVATION=0,MXFP8_GROUP_HINT=1024,SCHED_LOOKUP=1,E2E_PRECISION='mxfp8',E2E_REQUEST_COUNT=128,E2E_ROUNDS=1,CONTROLLED_GC=1,SHORTCUT_OVERLAP=1,E2E_MODEL_GRAPHS=1,E2E_ROUTED_GRAPHS=0,HYBRID_MCORE_ROOT=str(mc),TANH_LOGIT_SOURCE=str(root),TEACHER_CORPUS=str(corpus_path)).items():
        env['NT4_'+k] = str(v)
    for key in ('CUDA_LAUNCH_BLOCKING','MEGA_USE_NCU'):
        env.pop(key,None)
    for arm in ('cutlass','megamoe-exp','megamoe-tanh','trtllm'):
        output = root/arm
        output.mkdir()
        current = env.copy()
        hw, trt = arm=='megamoe-tanh', arm=='trtllm'
        current.update(NT4_E2E_OUTPUT=str(output),NT4_E2E_MODE='baseline' if arm=='cutlass' else 'direct',NT4_MXFP8_NVLS_HYBRID=str(int(arm!='cutlass')),NT4_TRTLLM_CLAMPED=str(int(trt)),NT4_TRT_EXPLORATORY_TIMING=str(int(trt)),NT4_VALIDATION_ISOLATE_TRT=str(int(trt)))
        for key in ('HARDWARE_TANH','TANH_FC1_ADVISORY','TANH_EXPERT_ADVISORY','TANH_OUTPUT_ADVISORY'):
            current['NT4_'+key] = str(int(hw))
        if arm=='cutlass':
            current['NT4_CUTLASS_DECODE_PROFILE_IDS'] = '[39,428]'
        (output/'command.json').write_text(json.dumps(command,indent=2))
        (output/'experiment-env.json').write_text(json.dumps({k:v for k,v in current.items() if k.startswith(('NT4_','FLASHINFER_'))},indent=2))
        print('Starting', arm, flush=True)
        with (root/(arm+'.log')).open('w') as stream:
            p = subprocess.Popen(command,env=current,cwd=mc,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc = p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL)
                    p.wait()
                rc = 124
        print('Completed', arm, 'exit', rc, flush=True)
        if rc:
            raise RuntimeError(f'{arm} failed; inspect {root}/{arm}.log')
    summarize(root)


if __name__ == '__main__':
    main()

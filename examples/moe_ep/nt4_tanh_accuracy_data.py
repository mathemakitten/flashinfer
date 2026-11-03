"""Select a tuning-disjoint labelled diagnostic sample with recorded provenance."""
import argparse, hashlib, json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=Path('/lustre/fsw/portfolios/nemotron/users/helenn')
    excluded={json.loads(x)['text'].strip() for x in (base/'research/routing/openmath2_unique2048.jsonl').read_text().splitlines()}
    directory=Path('/lustre/fsw/portfolios/nemotron/users/rkirby/hf_home/datasets/nvidia___open_math_instruct-2/default/0.0.0/469216e3f46f4dacf476b382e192485ea51a143e')
    # One bounded sequential shard scan, not repeated filesystem searches.
    shard=directory/'open_math_instruct-2-train_5M-00000-of-00014.arrow'
    seen={};conflicts=set();n=0
    with pa.memory_map(str(shard),'r') as f:
        reader=ipc.open_stream(f)
        for batch in reader:
            cols=batch.select(['problem','expected_answer','problem_source']).to_pydict()
            for question,answer,source in zip(cols['problem'],cols['expected_answer'],cols['problem_source']):
                n+=1
                if not question or not answer:continue
                question=question.strip()
                if question in excluded or len(question)>4000:continue
                if question in seen and seen[question]['expected_answer']!=answer:conflicts.add(question)
                seen.setdefault(question,dict(problem=question,expected_answer=answer,problem_source=source,source_row=n-1))
    choices=[v for k,v in seen.items() if k not in conflicts]
    choices.sort(key=lambda x:hashlib.sha256(('nt4-tanh-qualification-v1:'+x['problem']).encode()).hexdigest())
    assert len(choices)>=512
    chosen=choices[:512]
    for i,row in enumerate(chosen):
        row['index']=i
        row['text']=row['problem']+'\n\nSolve the problem step by step. Put your final answer in '+chr(92)+'boxed{}.'
    # Labels are separate from the prompt files consumed by generation.
    (a.output/'labels.json').write_text(json.dumps(chosen,indent=2))
    for batch in range(4):
        (a.output/f'prompts{batch}.jsonl').write_text(''.join(json.dumps({'text':r['text']})+'\n' for r in chosen[batch*128:(batch+1)*128]))
    command=json.loads((base/'logs/nt4-tanh-trt512-trt-first-20261030-174251/baseline-start-command.json').read_text())
    tokenizer=command[command.index('--tokenizer-model')+1]
    (a.output/'manifest.json').write_text(json.dumps(dict(source=str(shard),scanned_rows=n,eligible_unique=len(choices),conflicting_labels=len(conflicts),count=512,seed='nt4-tanh-qualification-v1',excluded_tuning_prompts=len(excluded),max_problem_characters=4000,tokenizer=tokenizer,limitations='One training-dataset shard; tuning-disjoint, not certified absent from model training. Dataset expected_answer labels; not generated_solution.'),indent=2))
    print(json.dumps(dict(stage='labelled_sample_ready',count=512,output=str(a.output))),flush=True)

if __name__=='__main__':main()

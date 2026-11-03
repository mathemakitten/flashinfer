"""Paired boxed-answer screening, with explicit coverage and truncation metrics."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from math_verify import parse, verify


def last_box(text):
    start=text.rfind('\\boxed{')
    if start<0:return None
    start+=7;depth=1
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i]
    return None


def wilson(success,n):
    if not n:return None
    z=1.95996398454;p=success/n;den=1+z*z/n
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    mid=(p+z*z/(2*n))/den
    return [mid-half,mid+half]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    labels=json.loads((a.data/'labels.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(json.loads((a.data/'manifest.json').read_text())['tokenizer'],local_files_only=True)
    eos=tokenizer.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    records=[];completed=[]
    for batch in range(4):
        folder=a.root/f'batch{batch}'
        names=('hybrid-control-start','hybrid-hardware-tanh','hybrid-control-end')
        files=[folder/name/'outputs-round1.json' for name in names]
        if not all(f.is_file() for f in files):continue
        arrays=[json.loads(f.read_text()) for f in files]
        assert all(len(x)==128 for x in arrays)
        if arrays[0]!=arrays[2]:raise AssertionError('Exponential control output changed across brackets')
        completed.append(batch)
        for i,(left,right) in enumerate(zip(arrays[0],arrays[1])):
            label=labels[batch*128+i];gold=parse('$'+label['expected_answer']+'$')
            row=dict(index=label['index'],problem_source=label['problem_source'],problem=label['problem'],expected_answer=label['expected_answer'],gold_parseable=bool(gold),exact_tokens=left==right)
            pos=next((j for j,(x,y) in enumerate(zip(left,right)) if x!=y),min(len(left),len(right)))
            if left!=right:
                row['first_difference']=dict(position=pos,common_prefix_ids=left[:pos],control_id=left[pos] if pos<len(left) else None,hardware_id=right[pos] if pos<len(right) else None,control_token=tokenizer.decode(left[pos:pos+1]),hardware_token=tokenizer.decode(right[pos:pos+1]))
            for name,tokens in [('exponential',left),('hardware',right)]:
                end=next((j for j,t in enumerate(tokens) if t in eos),len(tokens))
                text=tokenizer.decode(tokens[:end],skip_special_tokens=True)
                answer=last_box(text);candidate=parse('$'+answer+'$') if answer is not None else []
                row[name]=dict(text=text,boxed_answer=answer,answer_parseable=bool(candidate),eos_seen=end<len(tokens),generated_length=len(tokens),correct=bool(verify(gold,candidate)) if gold and candidate else False)
            records.append(row)
    if not records:raise RuntimeError('No complete paired generation batches')
    eligible=[r for r in records if r['gold_parseable']]
    n=len(eligible);report=dict(completed_batches=completed,total_requests=len(records),gold_parseable=n,unparseable_gold=len(records)-n,exact_token_requests=sum(r['exact_tokens'] for r in records),qualification='SCREENING_ONLY; not a shipping acceptance decision',scope='512 tuning-disjoint questions from one OpenMath training shard; first EOS or 2048-token limit; last complete boxed answer; unparseable/missing candidate answers scored incorrect, unparseable gold excluded explicitly')
    for arm in ('exponential','hardware'):
        count=sum(r[arm]['correct'] for r in eligible)
        report[arm]=dict(correct=count,total=n,accuracy=count/n if n else None,wilson95=wilson(count,n),missing_or_unparseable_answer=sum(not r[arm]['answer_parseable'] for r in eligible),no_eos_before_limit=sum(not r[arm]['eos_seen'] for r in eligible))
    diff=np.array([int(r['hardware']['correct'])-int(r['exponential']['correct']) for r in eligible])
    if n:
        rng=np.random.default_rng(20261102)
        boot=np.array([rng.choice(diff,n,replace=True).mean() for _ in range(10000)])
        report['paired_difference']=dict(hardware_minus_exp=float(diff.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),hardware_only_correct=int((diff==1).sum()),exponential_only_correct=int((diff==-1).sum()))
    (a.root/'answer-score-summary.json').write_text(json.dumps(report,indent=2))
    (a.root/'answer-score-records.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Launch a single native CUTLASS base-model probe with true EOS stopping."""
import datetime,json,os,subprocess,sys
from pathlib import Path
b=Path('/lustre/fsw/portfolios/nemotron/users/helenn');fi=b/'flashinfer-nt4-gb300-nvls';mc=b/'megatron-lm-nt4-coscheduling'
r=b/('logs/nt4-base-prompt-probe-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));r.mkdir()
source=b/'logs/nt4-tanh-answer-screen-20261102-074414'
labels=json.loads((source/'data/labels.json').read_text())
qs=['What is 2 + 2?',labels[0]['problem'],labels[3]['problem'],labels[56]['problem']]
fewshot='Question: What is 3 + 5?\nAnswer: 3 + 5 = 8. The answer is 8.\n\nQuestion: A rectangle has length 6 and width 4. What is its area?\nAnswer: The area is length times width, 6 * 4 = 24. The answer is 24.\n\nQuestion: Solve x + 7 = 12.\nAnswer: Subtract 7 from both sides: x = 5. The answer is 5.\n\n'
records=[]
for i,q in enumerate(qs):
 raw=q+'\n\nSolve the problem step by step. Put your final answer in '+chr(92)+'boxed{}.'
 for name,text in [('original',raw),('newline',raw+'\n'),('qa','Question: '+q+'\nAnswer:'),('fewshot',fewshot+'Question: '+q+'\nAnswer:')]:
  records.append(dict(question=i,format=name,text=text))
(r/'prompts.jsonl').write_text(''.join(json.dumps(dict(text=x['text']))+'\n' for x in records))
(r/'prompt-manifest.json').write_text(json.dumps(records,indent=2))
cmd=json.loads((source/'batch0/baseline-start-command.json').read_text());cmd[0]=sys.executable;cmd[5]=str(fi/'examples/moe_ep/nt4_base_prompt_probe.py')
def setarg(key,value):
 if key in cmd:cmd[cmd.index(key)+1]=str(value)
 else:cmd.extend([key,str(value)])
for k,v in {'--prompt-file':r/'prompts.jsonl','--prompt-file-num-truncate':16,'--num-tokens-to-generate':128,'--termination-id':2,'--cuda-graph-impl':'none','--inference-cuda-graph-scope':'none','--inference-dynamic-batching-num-cuda-graphs':0}.items():setarg(k,v)
if '--decode-only-cuda-graphs' in cmd:cmd.remove('--decode-only-cuda-graphs')
env={k:v for k,v in os.environ.items() if not k.startswith('NT4_')}
env.update(PYTHONPATH=f'{fi}:{mc}',FLASHINFER_WORKSPACE_BASE=str(b),MAX_JOBS='4',NT4_ALLOW_SOURCE_JIT='1',NT4_TE_STREAM_WORKSPACES='1',NT4_BASE_PROBE_OUTPUT=str(r/'outputs.json'))
(r/'command.json').write_text(json.dumps(cmd,indent=2))
print('Results:',r,flush=True)
with (r/'probe.log').open('w') as f:
 process=subprocess.Popen(cmd,env=env,cwd=mc,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=process.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  import signal
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
  code=124
print('Probe exit:',code,flush=True)
sys.exit(code)

"""Small base-model prompt/EOS probe; no accuracy or timing qualification."""
import json,os,runpy
from pathlib import Path
import torch
import torch.distributed as dist
from examples.inference.advanced import gpt_dynamic_inference as generation

original=generation.run_inference

def probe(requests,engine,*args,**kwargs):
    result=original(requests,engine,*args,**kwargs)
    if dist.get_rank()==0:
        tokenizer=engine.controller.tokenizer
        rows=[]
        for req in requests:
            tokens=req.output_tokens
            if isinstance(tokens,torch.Tensor):tokens=tokens.cpu().tolist()
            rows.append(dict(prompt=req.prompt_text,prompt_ids=req.prompt_tokens,
                prompt_contains_eos=tokenizer.eod in req.prompt_tokens,
                eos_id=tokenizer.eod,output_ids=tokens,
                output_text=tokenizer.detokenize(tokens),
                first_token_is_eos=bool(tokens and tokens[0]==tokenizer.eod)))
        output=Path(os.environ['NT4_BASE_PROBE_OUTPUT'])
        output.write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(stage='base_prompt_probe_saved',output=str(output),requests=len(rows))),flush=True)
    return result

generation.run_inference=probe
runpy.run_path(str(Path(__file__).with_name('nt4_checkpoint_generate.py')),run_name='__main__')

"""Diagnostic common-prefix logits with all requests teacher-forced.

No throughput or generation qualification claims: forced token equality is
bookkeeping validation, not evidence that the model selected the same tokens.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from nt4_capture_state import set_context
from nt4_gc_policy import controlled_gc


def run_probe(engine, args, tokenizer, sampling, build_requests, run_inference,
              adapter, model_session, destination):
    if model_session is None or engine.controller.num_speculative_tokens:
        raise ValueError('Logit probe requires whole-model graphs and no MTP')
    if args.top_k != 1:
        raise ValueError('Logit attribution requires greedy top_k=1')
    source = Path(os.environ['NT4_TANH_LOGIT_SOURCE'])
    reference = json.loads((source/'hybrid-control-start/outputs-round1.json').read_text())
    report = json.loads((source/'tanh-output-comparison.json').read_text())
    candidates = sorted(report['comparisons'][0]['differences'], key=lambda x:x['first_difference'])[:2]
    targets = {x['request_index']:x['first_difference'] for x in candidates}
    c = engine.controller
    ctx = engine.context
    original = c._dynamic_step_sample_logits
    data = {}
    # Arm identity is explicit: no graph cache or workspace is reused across arms.
    print(json.dumps(dict(stage='tanh_logit_probe_start', targets=targets,
                         performance_use=False, teacher_forced_all_requests=True)),flush=True)
    try:
        for round_id in (-1, 1):
            if round_id == 1:
                model_session.build()
            model_session.begin_round(round_id == 1)
            torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
            engine.reset()
            requests = build_requests(args,tokenizer,sampling)
            if len(requests)!=len(reference):
                raise ValueError('Request count differs from reference generation')
            positions = [0]*len(requests)
            found = {}; schedule = hashlib.sha256()
            def sample(*aa, **kw):
                n = ctx.total_request_count-ctx.paused_request_count
                ids = ctx.request_ids[ctx.paused_request_count:ctx.total_request_count].cpu().tolist()
                chunk = int(ctx.chunked_prefill_request_id)
                if len(ids)!=n or len(set(ids))!=n or any(i<0 or i>=len(reference) for i in ids):
                    raise ValueError('Unexpected request mapping')
                schedule.update(json.dumps([ids,[positions[i] for i in ids],chunk]).encode())
                logits = c._all_logits_cuda.squeeze(0)
                gathered = None if ctx.config.materialize_only_last_token_logits else ctx.gpu_view.active_request_last_token_idxs
                captures = []
                for row,rid in enumerate(ids):
                    if rid!=chunk and rid in targets and positions[rid]==targets[rid]:
                        logit_row = row if gathered is None else int(gathered[row])
                        vector = logits[logit_row].detach().float().cpu().clone()
                        if not bool(torch.isfinite(vector).all()):
                            raise ValueError('Nonfinite target logits')
                        captures.append((rid,vector))
                result=original(*aa,**kw)
                sampled = c._sampled_tokens_cuda[:n].cpu().tolist()
                forced = list(sampled)
                for row,rid in enumerate(ids):
                    if rid==chunk:continue
                    pos=positions[rid]
                    if pos>=len(reference[rid]):raise ValueError('Generation exceeds saved reference')
                    forced[row]=reference[rid][pos]
                    positions[rid]+=1
                c._sampled_tokens_cuda[:n].copy_(torch.tensor(forced,device=c._sampled_tokens_cuda.device,dtype=c._sampled_tokens_cuda.dtype))
                for rid,vector in captures:
                    row=ids.index(rid);pos=targets[rid]
                    topv,topi=vector.topk(10)
                    found[rid]=dict(request_index=rid,generated_position=pos,
                        prompt_text=requests[rid].prompt_text,
                        prompt_token_ids=tokenizer.tokenize(requests[rid].prompt_text),
                        common_generated_prefix_ids=reference[rid][:pos],
                        common_generated_prefix_text=tokenizer.detokenize(reference[rid][:pos]),
                        raw_argmax_id=int(topi[0]),sampled_before_override=int(sampled[row]),
                        forced_next_id=reference[rid][pos],
                        top_two_margin=float(topv[0]-topv[1]),
                        top_tokens=[dict(id=int(i),logit=float(v),text=tokenizer.detokenize([int(i)])) for i,v in zip(topi,topv)],
                        schedule_sha256_at_capture=schedule.hexdigest(),logits=vector)
                return result
            c._dynamic_step_sample_logits=sample
            set_context(ctx)
            try:
                with controlled_gc(True):run_inference(requests,engine)
            finally:
                c._dynamic_step_sample_logits=original
                set_context(None)
            torch.cuda.synchronize()
            observed=[r.output_tokens.cpu().tolist() if isinstance(r.output_tokens,torch.Tensor) else r.output_tokens for r in requests]
            if observed!=reference or set(found)!=set(targets):
                raise AssertionError('Teacher-forced output or target capture incomplete')
            data[round_id]=found
            if dist.get_rank()==0:
                torch.save(found,destination/f'logits-round{round_id}.pt')
                (destination/f'logits-round{round_id}.json').write_text(json.dumps({str(k):{key:v for key,v in x.items() if key!='logits'} for k,x in found.items()},indent=2))
            print(json.dumps(dict(stage='tanh_logit_probe_round_complete',rank=dist.get_rank(),round=round_id,
                                 targets=len(found),forced_tokens_match=True,performance_use=False)),flush=True)
        # Compare logits themselves, not the forced output IDs.
        checks={str(r):dict(eager_graph_exact=torch.equal(data[-1][r]['logits'],data[1][r]['logits']),
                           max_abs=float((data[-1][r]['logits']-data[1][r]['logits']).abs().max())) for r in targets}
        if dist.get_rank()==0:(destination/'logit-replay-check.json').write_text(json.dumps(checks,indent=2))
    finally:
        c._dynamic_step_sample_logits=original
        model_session.close()
        adapter.close()

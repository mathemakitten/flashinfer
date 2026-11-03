"""Paired conditional likelihood diagnostic; never a throughput benchmark."""
import hashlib
import json
import os
from pathlib import Path


def run_probe(engine, args, tokenizer, sampling, build_requests, run_inference,
              adapter, model_session, destination):
    import torch
    import torch.distributed as dist
    from nt4_capture_state import set_context
    from nt4_gc_policy import controlled_gc

    corpus = json.loads(Path(os.environ['NT4_TEACHER_CORPUS']).read_text())
    docs = corpus['documents']
    if model_session is None or engine.controller.num_speculative_tokens:
        raise ValueError('Requires model graphs without speculative decoding')
    controller, ctx = engine.controller, engine.context
    original = controller._dynamic_step_sample_logits
    results = {}
    try:
        for round_id in (-1, 1):
            if round_id == 1:
                model_session.build()
            model_session.begin_round(round_id == 1)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            engine.reset()
            requests = build_requests(args, tokenizer, sampling)
            if len(requests) != len(docs):
                raise ValueError('Corpus/request count mismatch')
            # Preserve exact token boundaries, without a decode/re-encode round trip.
            for request, doc in zip(requests, docs):
                request.prompt_tokens = list(doc['prefix_ids'])
            positions = [0] * len(docs)
            records = [[] for _ in docs]
            schedule = hashlib.sha256()

            def sample(*aa, **kw):
                ids = ctx.request_ids[ctx.paused_request_count:ctx.total_request_count].cpu().tolist()
                chunk = int(ctx.chunked_prefill_request_id)
                if len(set(ids)) != len(ids) or any(i < 0 or i >= len(docs) for i in ids):
                    raise ValueError('Unexpected request mapping')
                schedule.update(json.dumps([ids, [positions[i] for i in ids], chunk]).encode())
                logits = controller._all_logits_cuda.squeeze(0)
                gathered = None if ctx.config.materialize_only_last_token_logits else ctx.gpu_view.active_request_last_token_idxs
                live = [(row, rid) for row, rid in enumerate(ids) if rid != chunk]
                if live:
                    rows = [row if gathered is None else int(gathered[row]) for row, _ in live]
                    if logits.shape[-1] < corpus['vocab_size']:
                        raise ValueError('Logit vocabulary is incomplete')
                    vectors = logits[rows, :corpus['vocab_size']].float()
                    if not bool(torch.isfinite(vectors).all()):
                        raise AssertionError('Nonfinite logits')
                    targets = torch.tensor([docs[rid]['target_ids'][positions[rid]] for _, rid in live], device=vectors.device)
                    logp = vectors.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(vectors, dim=-1)
                    top = vectors.topk(2, dim=-1)
                    values = torch.stack([logp, top.indices[:, 0].float(), top.values[:, 0]-top.values[:, 1]], dim=1).cpu().tolist()
                    for (_, rid), (lp, best, margin) in zip(live, values):
                        records[rid].append(dict(position=positions[rid], target=docs[rid]['target_ids'][positions[rid]], logprob=lp, top_id=int(best), margin=margin))
                result = original(*aa, **kw)
                forced = controller._sampled_tokens_cuda[:len(ids)].clone()
                for row, rid in live:
                    forced[row] = docs[rid]['target_ids'][positions[rid]]
                    positions[rid] += 1
                controller._sampled_tokens_cuda[:len(ids)].copy_(forced)
                return result

            controller._dynamic_step_sample_logits = sample
            set_context(ctx)
            try:
                with controlled_gc(True):
                    run_inference(requests, engine)
            finally:
                controller._dynamic_step_sample_logits = original
                set_context(None)
            torch.cuda.synchronize()
            observed = [r.output_tokens.cpu().tolist() if isinstance(r.output_tokens, torch.Tensor) else list(r.output_tokens) for r in requests]
            if observed != [d['target_ids'] for d in docs]:
                raise AssertionError('Teacher forcing did not reproduce the corpus')
            if positions != [len(d['target_ids']) for d in docs]:
                raise AssertionError('Incomplete likelihood capture')
            payload = dict(corpus_sha256=hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(), schedule_sha256=schedule.hexdigest(), documents=records, forced_tokens_exact=True, performance_use=False)
            # All EP ranks must have observed the same scores and ordering.
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            digests = [None] * dist.get_world_size()
            dist.all_gather_object(digests, digest)
            if len(set(digests)) != 1:
                raise AssertionError('Rank likelihood records differ')
            results[round_id] = payload
            if dist.get_rank() == 0:
                (destination / f'likelihood-round{round_id}.json').write_text(json.dumps(payload))
            print(json.dumps(dict(stage='teacher_likelihood_complete', rank=dist.get_rank(), round=round_id, tokens=sum(positions))), flush=True)
        left, right = results[-1], results[1]
        errors = [abs(a['logprob']-b['logprob']) for da, db in zip(left['documents'], right['documents']) for a, b in zip(da, db)]
        check = dict(eager_graph_exact=left == right, max_logprob_difference=max(errors), same_schedule=left['schedule_sha256']==right['schedule_sha256'])
        if dist.get_rank() == 0:
            (destination/'replay-check.json').write_text(json.dumps(check, indent=2))
        if not check['eager_graph_exact']:
            raise AssertionError('Eager/graph likelihood differs; inspect replay-check.json')
    finally:
        controller._dynamic_step_sample_logits = original
        model_session.close()
        adapter.close()


if __name__ == '__main__':
    import nt4_tanh_logit_probe
    nt4_tanh_logit_probe.run_probe = run_probe
    from nt4_trt_validation_isolation import main
    main()

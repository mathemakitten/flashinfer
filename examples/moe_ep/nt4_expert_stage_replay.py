"""Standalone FC1/FC2 attribution on a captured local expert workload.

Diagnostic only: no NVLS/mixer; TRT PDL is disabled. MegaMoE uses its lean
compute base, not the communicating wrapper. Stage filtering retains scheduler
traversal overhead. These timings must not be called an E2E speedup.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import statistics


def checked(result):
    if int(result[0]):
        raise RuntimeError(repr(result))
    return result[1] if len(result) == 2 else result[1:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--iterations', type=int, default=100)
    sweep = p.add_mutually_exclusive_group()
    sweep.add_argument('--fc2-scheduling-sweep', action='store_true')
    sweep.add_argument('--fc2-pipeline-sweep', action='store_true')
    sweep.add_argument('--fc2-experiment', choices=('atomic','geometry','geometry_atomic','clc'))
    sweep.add_argument('--fc12-confirm', action='store_true')
    sweep.add_argument('--fc12-lookup', action='store_true')
    sweep.add_argument('--fc12-mixed-geometry', action='store_true')
    sweep.add_argument('--fc12-release-sweep', action='store_true')
    sweep.add_argument('--fc12-store-sweep', action='store_true')
    sweep.add_argument('--fc12-probe-sweep', action='store_true')
    p.add_argument('--workers',type=int,default=152)
    p.add_argument('--trt-only', action='store_true')
    p.add_argument('--trt-fc2-config', type=int, default=7)
    p.add_argument('--trt-fc1-config', type=int, default=5)
    p.add_argument('--trt-export', type=Path)
    p.add_argument('--expected-scheduler', choices=('static','static-persistent','clc'))
    a = p.parse_args()
    if a.fc12_release_sweep or a.fc12_store_sweep or a.fc12_probe_sweep: a.fc12_mixed_geometry = True
    if a.fc12_mixed_geometry or a.fc12_lookup: a.fc12_confirm = True
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ['NT4_TRT_DEBUG_INTERMEDIATES'] = '1'
    os.environ['NT4_EXPERT_INTERNAL_TIMING'] = '0'
    os.environ['NT4_EXPERT_PHASE_TIMING'] = '0'
    os.environ['NT4_CUBIN_TRACE'] = '0'
    os.environ.pop('NT4_EXPERT_STAGE_CAPTURE', None)
    import torch
    import cuda.bindings.driver as driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from nt4_trtllm_diagnostic import metrics, tensor_hash, intermediate_sf_layout
    saved = torch.load(a.artifact / 'inputs.pt', map_location='cpu', weights_only=True)
    assert saved['schema'] == 'nt4_expert_stage_v1' and saved['layer'] == 32
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    export = str(a.trt_export.resolve()) if a.trt_export else saved['export']
    os.environ['FLASHINFER_TRTLLM_NT4_EXPORT'] = export
    from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a, mxfp8_quantize
    import flashinfer.fused_moe as fm
    from flashinfer.fused_moe import nt4_diagnostic as tap
    from nt4_trtllm_surrogate import TrtllmNvlsSurrogate
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.nt4.kernel_mxfp8_glu_fc12 import Sm100ClampedRelu2Mxfp8Fc12Kernel as Kernel
    from moe_nvfp4_swapab.runner_common import to_blocked, from_blocked, assemble_raw_scales_grouped_token, assemble_raw_scales_stacked_expert

    fm.trtllm_fp8_block_scale_routed_moe = functools.partial(fm.trtllm_fp8_block_scale_routed_moe, enable_pdl=False)
    h = object.__new__(TrtllmNvlsSurrogate)
    h.rank, h.tuned, h.surrogate = saved['rank'], False, False
    h.x, h.ids, h.probs = [saved[k].cuda() for k in ('x','ids','probs')]
    h.clamp_limit = saved['clamp_limit'].cuda()
    w = {k:v.cuda() for k,v in saved['weights'].items()}
    h.weights = {}
    for wk,sk in [('w1','s1'),('w2','s2')]:
        h.weights[wk] = torch.stack([shuffle_matrix_a(t.view(torch.uint8),128) for t in w[wk]]).view(torch.float8_e4m3fn)
        h.weights[sk] = torch.stack([shuffle_matrix_sf_a(t.view(torch.uint8),128) for t in w[sk]]).reshape(w[sk].shape)
    # Private standalone replacement: retain workspace ownership during capture,
    # without the ordinary eager diagnostic's CPU analysis or graph prohibition.
    captured = []
    def retain(buffers, tactic):
        assert len(buffers) == 5
        captured[:] = [v if v is None or isinstance(v,torch.Tensor) else torch.from_dlpack(v) for v in buffers]
    tap.record = retain
    # All four ranks in the matching serial capture selected FC configs 5/7.
    from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner
    original_init = MoERunner.__init__
    def pinned_init(runner, *args, **kwargs):
        original_init(runner, *args, **kwargs)
        legal = runner.moe_op.trtllm_get_valid_moe_factorizations(
            runner.dtype_act, runner.dtype_weights, runner.fp8_quantization_type,
            runner.top_k, runner.hidden_size, runner.hidden_size_output,
            runner.intermediate_size, runner.num_local_experts,
            runner.activation_type, runner.use_shuffled_weight, runner.weight_layout,
            runner.use_per_token_scaling, len(h.x), False)
        matches = [[int(row[0]), int(row[1])] for row in legal
                   if int(row[2]) == a.trt_fc1_config and int(row[3]) == a.trt_fc2_config]
        assert len(matches) == 1, matches
        tap.forced_tactic = matches[0]
    MoERunner.__init__ = pinned_init
    h.local()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        full_output = h.local()
    graph.instantiate()
    graph.replay()
    torch.cuda.synchronize()
    q1,s1,y2,mapping,_ = captured
    q1_seed,s1_seed,y2_seed = [v.clone() for v in (q1,s1,y2)]
    mapping = mapping.reshape(h.ids.shape).long()
    raw_graph = driver.CUgraph(graph.raw_cuda_graph())
    _,count = checked(driver.cuGraphGetNodes(raw_graph,0))
    nodes,count = checked(driver.cuGraphGetNodes(raw_graph,count))
    inventory=[]
    for node in nodes[:count]:
        kind = checked(driver.cuGraphNodeGetType(node))
        name = str(kind)
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            name = checked(driver.cuFuncGetName(checked(driver.cuGraphKernelNodeGetParams(node)).func)).decode()
        elif kind not in (driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY,driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET):
            raise RuntimeError(f'Unsupported graph node {kind}; do not time a partial graph')
        inventory.append((node,name))
    fc1_nodes=[n for n,name in inventory if name.startswith('bmm_') and 'clampedrelu2' in name]
    fc2_nodes=[n for n,name in inventory if name.startswith('bmm_') and 'Bfloat16' in name]
    assert len(fc1_nodes)==len(fc2_nodes)==1, [name for _,name in inventory]
    fc2_name = next(name for node,name in inventory if node in fc2_nodes)
    if a.expected_scheduler:
        expected = {'static': '_schedS_', 'static-persistent': '_schPs', 'clc': '_schPd'}[a.expected_scheduler]
        assert expected in fc2_name, (a.expected_scheduler,fc2_name)
    executable=driver.CUgraphExec(graph.raw_cuda_graph_exec())
    def trt_stage(stage):
        keep=fc1_nodes if stage==1 else fc2_nodes
        for node,_ in inventory:
            checked(driver.cuGraphNodeSetEnabled(executable,node,int(node in keep)))
        return graph.replay

    counts=[]; tokens=[]; perms=[]
    for e in range(128):
        tok,slot=torch.where(h.ids==h.rank*128+e)
        counts.append(len(tok)); tokens.append(tok); perms.append(mapping[tok,slot])
    assert sum(counts)>0
    layout=intermediate_sf_layout(export)
    def raw_sf(raw,rows,width):
        k=torch.arange(width//32,device='cuda')[None,:]; r=rows[:,None]
        if layout=='128x4': off=((r//128)*((width//32+3)//4)+k//4)*512+(r%32)*16+((r%128)//32)*4+k%4
        elif layout=='8x4': off=((r//8)*((width//32+3)//4)+k//4)*32+(r%8)*4+k%4
        elif layout=='linear': off=r*(width//32)+k
        else: raise ValueError(layout)
        return raw.view(torch.uint8).flatten()[off]
    qx,sx=mxfp8_quantize(h.x,False,32)
    sx=sx.view(torch.uint8).reshape(len(h.x),-1)
    offsets=[0]
    for n in counts: offsets.append(offsets[-1]+((n+63)//64)*64)
    rows=offsets[-1]
    fp8=torch.float8_e4m3fn; sf8=torch.float8_e8m0fnu
    x=torch.zeros((rows,768),device='cuda',dtype=torch.uint8).view(fp8)
    common=torch.zeros((rows,4096),device='cuda',dtype=torch.uint8).view(fp8)
    common_sf=[]; x_sf=[]; valid=[]; refs=[]; trt_terms=[]
    for e,n in enumerate(counts):
        sl=slice(offsets[e],offsets[e]+n)
        x.view(torch.uint8)[sl]=qx.view(torch.uint8)[tokens[e]]
        common.view(torch.uint8)[sl]=q1_seed.view(torch.uint8)[perms[e]]
        bits=raw_sf(s1_seed,perms[e],4096)
        common_sf.append(bits.view(sf8)); x_sf.append(sx[tokens[e]].view(sf8))
        valid.extend(range(offsets[e],offsets[e]+n))
        if n:
            act=common[sl].double()*torch.exp2(bits.double()-127).repeat_interleave(32,-1)
            down=w['w2'][e].view(fp8).double()*torch.exp2(w['s2'][e].view(torch.uint8).double()-127).repeat_interleave(32,-1)
            refs.append(act@down.T)
            trt_terms.append(y2_seed[perms[e]])
    idx=torch.tensor(valid,device='cuda'); ref=torch.cat(refs)
    common_scales=assemble_raw_scales_grouped_token(common_sf)
    def deblocked(scales):
        result=[]; pos=0
        for n in counts:
            size=((n+127)//128)*128*128
            if n: result.append(from_blocked(scales.view(torch.uint8).flatten()[pos:pos+size],n,128))
            pos+=size
        return torch.cat(result)
    assert torch.equal(deblocked(common_scales),torch.cat([v.view(torch.uint8) for v in common_sf]))
    fc2_params = checked(driver.cuGraphKernelNodeGetParams(fc2_nodes[0]))
    launch = {k: int(getattr(fc2_params, k)) for k in
              ('gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes')}
    report=dict(export=export, fc2_launch=launch, scope=__doc__,rank=h.rank,layer=32,counts=counts,
                source_branch=saved['branch_record'],tactic=tap.forced_tactic,
                tactic_policy=f'FC1 config{a.trt_fc1_config} and FC2 config{a.trt_fc2_config} pinned; PDL disabled for isolation',
                graph_nodes=[name for _,name in inventory],
                common_activation_sha256=tensor_hash(common.view(torch.uint8)[idx]),
                common_scale_sha256=tensor_hash(deblocked(common_scales)),
                fc2_trt_reference=metrics(torch.cat(trt_terms),ref),timings={})
    def numerical_gate(actual,label):
        m=metrics(actual,ref)
        report[label]=m
        if not m['finite'] or m['global_rel_l2']>.015 or m['max_row_rel_l2']>.015:
            (a.output/'failure.json').write_text(json.dumps(report,indent=2))
            raise AssertionError(f'{label}: provisional 1.5% independent gate failed: {m}')
    numerical_gate(torch.cat(trt_terms),'trt_fc2_independent')
    def bench(fn,label):
        for _ in range(20): fn()
        torch.cuda.synchronize()
        times=[]
        for _ in range(5):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.iterations):fn()
            end.record();end.synchronize()
            times.append(start.elapsed_time(end)*1000/a.iterations)
        report['timings'][label]=dict(rounds_us=times,median_us=statistics.median(times))
        print(json.dumps(dict(stage='isolated_timing',label=label,**report['timings'][label])),flush=True)
    for stage in (1,2):
        fn=trt_stage(stage)
        if stage==2:
            q1.copy_(q1_seed);s1.copy_(s1_seed);y2.fill_(float('nan'))
        fn();torch.cuda.synchronize()
        if stage==1:
            assert torch.equal(q1.view(torch.uint8),q1_seed.view(torch.uint8)), 'TRT FC1 graph isolation changed data'
        else:
            assert torch.equal(y2[torch.cat(perms)],y2_seed[torch.cat(perms)]), 'TRT FC2 isolation changed output'
        bench(fn,f'trt_fc{stage}')

    if a.trt_only:
        report['fc2_output_sha256']=tensor_hash(torch.cat(trt_terms))
        report['status']='PASS_TRT_STAGE_REPLAY'
        (a.output/'summary.json').write_text(json.dumps(report,indent=2))
        print('Complete:',a.output,flush=True)
        return

    output=torch.empty((rows,768),device='cuda',dtype=torch.bfloat16)
    fc1=torch.zeros_like(common.view(torch.uint8)).view(fp8)
    fc1sf=torch.zeros_like(common_scales.view(torch.uint8)).view(sf8)
    counters=torch.zeros((rows//128+256,),device='cuda',dtype=torch.int32)
    tensors=dict(activation=x,fc1_weight=w['w1'].view(fp8).permute(0,2,1),
        activation_sf=assemble_raw_scales_grouped_token(x_sf),
        fc1_weight_sf=assemble_raw_scales_stacked_expert([v.view(sf8) for v in w['s1']]),
        fc1_output=fc1,fc1_output_sf=fc1sf,fc2_weight=w['w2'].view(fp8).permute(0,2,1),
        fc2_weight_sf=assemble_raw_scales_stacked_expert([v.view(sf8) for v in w['s2']]),
        fc2_output=output,topk_scores=torch.ones(rows,device='cuda'),fc1_done_counter=counters,
        offs=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32))
    runtime={k:ct.from_dlpack(v,assumed_align=16).mark_layout_dynamic(leading_dim=ct.get_leading_dim(v)) for k,v in tensors.items()}
    runtime['stream']=driver.CUstream(torch.cuda.current_stream().cuda_stream)
    functions={}
    for stage in (0,1,2):
        kernel=Kernel(mma_tiler_mnk=(128,256,128),cluster_shape_mnk=(1,1,1),
            use_2cta_instrs=False,group_hint=512,token_padding_block=64,sf_padding_block=128,
            ab_dtype=cutlass.Float8E4M3FN,gate_up_clamp=16.,diagnostic_stage=stage)
        kernel.hardware_tanh=True
        compiled=cute.compile(kernel,**runtime,max_active_clusters=a.workers)
        def run(compiled=compiled):
            launch = dict(runtime)
            launch['stream'] = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled(**launch)
        functions[stage]=run
    counters.zero_();functions[0]();torch.cuda.synchronize()
    both_fc1=fc1.clone();both_sf=fc1sf.clone();both_out=output[idx].clone()
    counters.zero_();functions[1]();torch.cuda.synchronize()
    assert torch.equal(fc1.view(torch.uint8)[idx],both_fc1.view(torch.uint8)[idx]), 'FC1 isolated/full mismatch'
    assert torch.equal(deblocked(fc1sf),deblocked(both_sf)), 'FC1 scales isolated/full mismatch'
    # First prove FC2 isolation reproduces the fused kernel's own handoff.
    counters.fill_(16);output.fill_(float('nan'));functions[2]();torch.cuda.synchronize()
    assert torch.equal(output[idx],both_out), 'FC2 isolated/full mismatch'
    # Then substitute the identical TRT post-activation bytes and scales.
    fc1.copy_(common);fc1sf.copy_(common_scales);counters.fill_(16)
    functions[2]();torch.cuda.synchronize()
    numerical_gate(output[idx],'mega_fc2_independent')
    report['fc2_cross_backend']=metrics(output[idx],torch.cat(trt_terms))
    snapshot=output[idx].clone();output.fill_(float('nan'));functions[2]();torch.cuda.synchronize()
    assert torch.equal(output[idx],snapshot), 'FC2 poisoned replay mismatch'
    if a.fc2_scheduling_sweep or a.fc2_pipeline_sweep or a.fc2_experiment or a.fc12_confirm:
        # All-ready FC2 never modifies these counters: initialize outside capture.
        # Compare kernel-only replay against the reset-inclusive original measurement.
        fc1.copy_(common); fc1sf.copy_(common_scales); counters.fill_(16)
        reset_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(reset_graph): counters.fill_(16)
        bench(reset_graph.replay, 'counter_fill_only')
        selected_stage = 0 if a.fc12_confirm else 2
        expected_output = both_out if a.fc12_confirm else snapshot
        candidates = [('control_start',512,functions[selected_stage])]
        report['workers']=a.workers
        report['mixed_fc2_geometry']=a.fc12_mixed_geometry
        report['poison_intermediates']=a.fc12_mixed_geometry
        report['selected_stage']=selected_stage
        variants = ([('ab_reduce1',512,1)] if a.fc2_pipeline_sweep else
                    [('group128',128,0),('group1024',1024,0)])
        variants = [(label,group,reduction,256,'static') for label,group,reduction in variants]
        if a.fc2_experiment == 'atomic':
            variants = [('atomic512',512,0,256,'atomic_counter'),('atomic1024',1024,0,256,'atomic_counter')]
        elif a.fc2_experiment == 'geometry':
            variants = [('n128',512,0,128,'static'),('n128_reduce1',512,1,128,'static')]
        if a.fc2_experiment == 'geometry_atomic':
            variants = [(f'n{n}_{mode}_g{group}',group,0,n,mode)
                        for n in (256,128) for group in (128,512,1024)
                        for mode in ('static','atomic_counter')]
            report['scheduling_scope'] = 'software atomic work claiming, not hardware CLC; FC2 all-ready'
        if a.fc2_experiment == 'clc':
            variants = [('n128_static',128,0,128,'static'),
                        ('n128_fc2_static',128,0,128,'fc2_static'),
                        ('n128_fc2_grid',128,0,128,'fc2_grid'),
                        ('n128_clc',128,0,128,'clc'),
                        ('n128_fc2_grid_end',128,0,128,'fc2_grid'),
                        ('n128_fc2_static_end',128,0,128,'fc2_static'),
                        ('n128_static_end',128,0,128,'static')]
            report['scheduling_scope'] = 'hardware CLC, one CTA per FC2 tile; order-independent FC2-only decoder with matched static control'
        if a.fc12_confirm:
            variants=[('group1024',1024,0,256,'static'),
                      ('atomic512',512,0,256,'atomic_counter'),
                      ('atomic1024',1024,0,256,'atomic_counter'),
                      ('ab_reduce1',512,1,256,'static')]
        if a.fc12_lookup:
            variants = [('group1024_control',1024,0,256,'static'),
                        ('group1024_lookup',1024,0,256,'static'),
                        ('group1024_control_end',1024,0,256,'static')]
        if a.fc12_mixed_geometry:
            variants=[('group1024_n256',1024,0,256,'static'),
                      ('group1024_fc2n128',1024,0,256,'static')]
        release_policies = {}
        if a.fc12_release_sweep:
            release_policies = {
                'group1024_n256_late': 'late',
                'group1024_fc2n128_disjoint_late': 'disjoint_late',
                'group1024_fc2n128_disjoint_early': 'disjoint_early',
            }
            variants = [(label,1024,0,256,'static') for label in (
                'group1024_n256', 'group1024_n256_late', 'group1024_fc2n128',
                'group1024_fc2n128_disjoint_late', 'group1024_fc2n128_disjoint_early')]
        if a.fc12_store_sweep:
            variants=[('group1024_n256',1024,0,256,'static'),
                      ('group1024_stg',1024,0,256,'static'),
                      ('group512_stg',512,0,256,'static')]
        if a.fc12_probe_sweep:
            variants = [(label,1024,0,256,'static') for label in (
                'probe_control_start', 'probe_after', 'probe_control_end')]
        report['fc2_probe_sweep'] = a.fc12_probe_sweep
        report['fc1_store_sweep'] = a.fc12_store_sweep
        report['fc2_release_sweep'] = a.fc12_release_sweep
        lb = torch.zeros(1,device='cuda',dtype=torch.int32)
        lb_arg = ct.from_dlpack(lb,assumed_align=16).mark_layout_dynamic(leading_dim=0)
        report['fc2_variants'] = [dict(label=label,group_hint=group,ab_stage_reduction=reduction,fc1_tile_n=n,fc2_tile_n=(128 if label.startswith('group1024_fc2n128') else n),fc2_acc_release=release_policies.get(label,'auto'),use_stg_fc1=label.endswith('_stg'),load_balance=mode)
                                  for label,group,reduction,n,mode in variants]
        for label,group,reduction,n,mode in variants:
            kernel=Kernel(mma_tiler_mnk=(128,n,128),cluster_shape_mnk=(1,1,1),
                use_2cta_instrs=False,group_hint=group,token_padding_block=64,sf_padding_block=128,
                ab_dtype=cutlass.Float8E4M3FN,gate_up_clamp=16.,diagnostic_stage=selected_stage,
                ab_stage_reduction=reduction,load_balance_mode=mode,
                static_expert_shape=(128,4096,768),
                scheduler_lookup=label.endswith('_lookup'),
                diagnostic_clc_tiles=(sum((c+127)//128 for c in counts) *
                    ((768+n-1)//n) if mode in ('clc','fc2_static','fc2_grid') else 0),
                fc2_tile_n=(128 if label.startswith('group1024_fc2n128') else None),
                fc2_acc_release=release_policies.get(label,'auto'),
                use_stg_fc1=label.endswith('_stg'))
            kernel.hardware_tanh=True
            kernel.fc2_probe_after_issue = label == 'probe_after'
            args=dict(runtime)
            if mode == 'atomic_counter': args['load_balance_counter']=lb_arg
            compile_options = {}
            dump_dir = None
            if mode == 'clc':
                import tempfile
                dump_dir = Path(tempfile.mkdtemp(prefix='nt4-clc-', dir='/tmp'))
                compile_options['options'] = f'--keep-ptx --keep-cubin --dump-dir={dump_dir}'
            compiled=cute.compile(kernel,**args,max_active_clusters=a.workers,**compile_options)
            if dump_dir is not None:
                ptx_files = list(dump_dir.rglob('*.ptx'))
                clc_lines = [line.strip() for path in ptx_files for line in path.read_text().splitlines()
                             if 'clusterlaunchcontrol.try_cancel' in line]
                assert clc_lines, 'Compiled CLC prototype contains no hardware cancellation instruction'
                report['clc_instruction_audit'] = dict(ptx_instructions=clc_lines,
                    dump_dir=str(dump_dir),grid=[1,1,kernel.diagnostic_clc_tiles],
                    static_workers=a.workers,clc_worker_cap=False)
                print(json.dumps(dict(stage='clc_instruction_audit',**report['clc_instruction_audit'])),flush=True)
            def run_fc2(compiled=compiled,args=args):
                launch=dict(args)
                launch['stream']=driver.CUstream(torch.cuda.current_stream().cuda_stream)
                compiled(**launch)
            candidates.append((label,group,run_fc2))
        if a.fc12_mixed_geometry:
            candidates.append(('group1024_n256_end',1024,candidates[1][2]))
        candidates.append(('control_end',512,functions[selected_stage]))
        if a.fc2_experiment:
            counters.fill_(32) # All-ready FC2, sufficient for either N128 or N256.
            report['counter_policy']='identical load-balance counter zero in every candidate graph'
        for label,group,raw_fn in candidates:
            def fn(raw_fn=raw_fn):
                if a.fc2_experiment or a.fc12_confirm: lb.zero_()
                if a.fc12_confirm: counters.zero_()
                raw_fn()
            if a.fc12_mixed_geometry or a.fc12_lookup:
                fc1.view(torch.uint8).fill_(0x7f)
                fc1sf.view(torch.uint8).fill_(0xff)
            output.fill_(float('nan'));fn();torch.cuda.synchronize()
            if not torch.equal(output[idx],expected_output):
                report['candidate_failure']=dict(label=label,
                    against_control=metrics(output[idx],expected_output),
                    against_common_fc2_reference=(None if a.fc12_confirm else metrics(output[idx],ref)),
                    nonfinite=int((~torch.isfinite(output[idx])).sum()),
                    differing=int((output[idx]!=expected_output).sum()))
                (a.output/'failure.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(report['candidate_failure']),flush=True)
                raise AssertionError(f'{label} eager mismatch')
            g=torch.cuda.CUDAGraph(keep_graph=a.fc2_experiment == 'clc')
            with torch.cuda.graph(g): fn()
            if a.fc2_experiment == 'clc':
                g.instantiate()
                graph_handle = driver.CUgraph(g.raw_cuda_graph())
                _, node_count = checked(driver.cuGraphGetNodes(graph_handle,0))
                graph_nodes, node_count = checked(driver.cuGraphGetNodes(graph_handle,node_count))
                launches = []
                for node in graph_nodes[:node_count]:
                    if checked(driver.cuGraphNodeGetType(node)) == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                        kp = checked(driver.cuGraphKernelNodeGetParams(node))
                        launches.append(dict(name=checked(driver.cuFuncGetName(kp.func)).decode(),
                            grid=[kp.gridDimX,kp.gridDimY,kp.gridDimZ],
                            block=[kp.blockDimX,kp.blockDimY,kp.blockDimZ],dynamic_smem=kp.sharedMemBytes,
                            registers=checked(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,kp.func))))
                report.setdefault('candidate_launches',{})[label] = launches
            if a.fc12_mixed_geometry or a.fc12_lookup:
                fc1.view(torch.uint8).fill_(0x7f)
                fc1sf.view(torch.uint8).fill_(0xff)
            output.fill_(float('nan'));g.replay();torch.cuda.synchronize()
            assert torch.equal(output[idx],expected_output), f'{label} graph mismatch'
            if a.fc12_confirm:
                assert torch.equal(fc1.view(torch.uint8)[idx],both_fc1.view(torch.uint8)[idx]), f'{label} FC1 mismatch'
                assert torch.equal(deblocked(fc1sf),deblocked(both_sf)), f'{label} FC1 scales mismatch'
            bench(g.replay, f'fc{12 if a.fc12_confirm else 2}_kernel_{label}')
            assert torch.equal(output[idx],expected_output), f'{label} post-timing output mismatch'
            if a.fc12_mixed_geometry or a.fc12_lookup:
                assert torch.equal(output[idx],expected_output), f'{label} post-timing mismatch'
                for poison in (0x00, 0x7f, 0xff):
                    fc1.view(torch.uint8).fill_(poison)
                    fc1sf.view(torch.uint8).fill_(0xff)
                    output.fill_(float('nan'))
                    g.replay(); torch.cuda.synchronize()
                    assert torch.equal(output[idx],expected_output), f'{label} repeated poisoned output mismatch'
                    assert torch.equal(fc1.view(torch.uint8)[idx],both_fc1.view(torch.uint8)[idx]), f'{label} repeated FC1 mismatch'
                    assert torch.equal(deblocked(fc1sf),deblocked(both_sf)), f'{label} repeated scales mismatch'
            (a.output/'partial.json').write_text(json.dumps(report,indent=2))
        # Re-measure the unchanged TRT kernel after the MegaMoE trials.
        bench(trt_stage(2),'trt_fc2_end_control')
        report['status']=('PASS_FC12_CONFIRM' if a.fc12_confirm else 'PASS_FC2_'+a.fc2_experiment.upper()+'_DIAGNOSTIC' if a.fc2_experiment else 'PASS_FC2_PIPELINE_DIAGNOSTIC' if a.fc2_pipeline_sweep
                          else 'PASS_FC2_SCHEDULING_DIAGNOSTIC')
        (a.output/'summary.json').write_text(json.dumps(report,indent=2))
        print('Complete:',a.output,flush=True)
        return
    for stage in (2,1,0):
        # Counters reset once per graph invocation; report this explicitly.
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if stage==2: counters.fill_(16)
            else: counters.zero_()
            functions[stage]()
        # Require execution from the graph, not stale outputs from an eager launch.
        if stage == 2: output.fill_(float('nan'))
        g.replay();torch.cuda.synchronize()
        if stage==2: assert torch.equal(output[idx],snapshot), 'FC2 graph/eager mismatch'
        bench(g.replay,f'mega_stage{stage}_including_counter_reset')
    report['status']='PASS_PROVISIONAL_FC2_AND_REPLAY_CHECKS'
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print('Complete:',a.output,flush=True)


if __name__=='__main__':
    main()

"""Isolated research bootstrap: validate TRT without executing MegaMoE first.

No shared harness files are changed. The resolved adapter source is saved beside
results for review. Existing numerical exception policy remains explicit.
"""
import os
import json
from pathlib import Path


def is_trt_backend(hybrid):
    from nt4_trtllm_surrogate import TrtllmNvlsSurrogate
    return isinstance(hybrid, TrtllmNvlsSurrogate) and hybrid.clamp_limit is not None


def check_trt_recombined(hybrid, actual, local_reference):
    import torch
    import torch.distributed as dist
    from nt4_checkpoint_replay import compare, emit
    if local_reference is None:
        raise AssertionError('Missing independent TRT reference')
    expected=local_reference.clone()
    dist.all_reduce(expected,group=hybrid.direct.group)
    start=hybrid.rank*hybrid.capacity
    expected=expected[start:start+hybrid.capacity]
    try:
        compare(actual.float(),expected,'trtllm_recombined_vs_independent_reference')
    except AssertionError as error:
        allowed=(os.environ.get('NT4_TRT_EXPLORATORY_TIMING')=='1'
                 and 'provisional global/per-token L2 gate failed' in str(error)
                 and bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                 and bool((actual[(expected==0).all(-1)]==0).all()))
        if not allowed:raise
        emit(stage='trtllm_recombined_numerical_exception',error=str(error),numerical_qualification='NOT_PASSED',status='EXPLORATORY_ONLY')
    emit(stage='backend_validation_selection',backend='TRTLLM',megamoe_prerequisite=False,
         checks='independent_local_and_recombined_reference; finite/zero; poisoned_routes; graph_replay')


def patched_source(source):
    substitutions=[
        ('if replay is not None and not checked:',
         'if replay is not None and not checked and not is_trt_backend(hybrid):'),
        ('                        hybrid.check_fused_helpers()',
         '                        independent_local = hybrid.check_fused_helpers()'),
        ('                        if not getattr(hybrid, "surrogate", False):',
         '                        if is_trt_backend(hybrid):\n                            check_trt_recombined(hybrid, eager_hybrid, independent_local)\n                        if not getattr(hybrid, "surrogate", False):'),
        ('                            compare(\n                                eager_hybrid, direct().clone(), "nvls_hybrid_vs_direct"\n                            )',
         '                            if not is_trt_backend(hybrid):\n                                compare(\n                                    eager_hybrid, direct().clone(), "nvls_hybrid_vs_direct"\n                                )'),
    ]
    for old,new in substitutions:
        if source.count(old)!=1:raise RuntimeError('Adapter changed: isolation patch needs review: '+old)
        source=source.replace(old,new,1)
    return source


def main():
    import hashlib
    import importlib.util
    import runpy
    import sys
    root=Path(__file__).resolve().parent
    source_path=root/'nt4_e2e_graph.py'
    source=source_path.read_text()
    if os.environ.get('NT4_VALIDATION_ISOLATE_TRT')=='1':
        patched=patched_source(source)
        output=Path(os.environ['NT4_E2E_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
        # One rank writes immutable provenance; no per-layer files.
        if os.environ.get('RANK','0')=='0':
            (output/'resolved_graph_adapter.py').write_text(patched)
            (output/'validation-provenance.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),resolved_sha256=hashlib.sha256(patched.encode()).hexdigest()),indent=2))
        spec=importlib.util.spec_from_file_location('nt4_e2e_graph',source_path)
        module=importlib.util.module_from_spec(spec)
        module.is_trt_backend=is_trt_backend
        module.check_trt_recombined=check_trt_recombined
        sys.modules['nt4_e2e_graph']=module
        exec(compile(patched,str(source_path),'exec'),module.__dict__)
    runpy.run_path(str(root/'nt4_checkpoint_generate.py'),run_name='__main__')

if __name__=='__main__':main()

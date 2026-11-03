"""Saved-input numerical audit; FP64 is a reference, not a proof or new gate."""
import argparse
import json
from pathlib import Path
import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from nt4_tanh_register_bench import Fragment
from nt4_expert_diagnostics import decode_e8m0_exact

def rel(x,y):
    return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))

def quant64(y):
    """Independent E8M0 ceiling + nearest-even E4M3 on CPU FP64 values."""
    b=y.reshape(-1,32).double()
    peak=b.abs().amax(1)/448
    frac,exp=torch.frexp(peak)
    power=torch.where(frac==0.5,exp-1,exp).clamp(-127,127)
    power=torch.where(peak==0,torch.full_like(power,-127),power)
    scale=torch.ldexp(torch.ones_like(peak),power)
    table=torch.arange(127,dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    v=(b/scale[:,None]).clamp(0,448)
    hi=torch.searchsorted(table,v.contiguous()).clamp(0,126)
    lo=(hi-1).clamp_min(0)
    dl=v-table[lo];dh=table[hi]-v
    idx=torch.where((dl<dh)|((dl==dh)&(lo%2==0)),lo,hi)
    return (table[idx]*scale[:,None]).reshape(y.shape),scale.reshape(*y.shape[:-1],-1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for path in a.snapshot:
        s=torch.load(path,map_location='cpu',weights_only=True)
        z64=s['x'].double()@s['up'].double().T
        target=(16*torch.tanh(z64.clamp_min(0)/16)).square()
        ideal,ideal_scale=quant64(target)
        actual=s['actual_fp8'].contiguous().view(torch.float8_e4m3fn).double()*decode_e8m0_exact(s['actual_scale_bytes']).repeat_interleave(32,-1)
        record=dict(snapshot=str(path),expert=int(s['expert']),recomputed_fc1_vs_saved_l2=rel(z64,s['z_reference_fp64']),sources=[])
        for source_name,z in [('fp32_reference',s['z_reference'].float()),('fp64_rounded_fp32',z64.float())]:
            own_target=(16*torch.tanh(z.double().clamp_min(0)/16)).square()
            own_q,own_sf=quant64(own_target)
            n=z.numel();size=((n+4095)//4096)*4096
            x=torch.zeros(size,device='cuda');x[:n].copy_(z.flatten())
            arms={}
            for hardware in (False,True):
                name='hardware' if hardware else 'exponential';values={}
                for quant in (False,True):
                    y=torch.empty_like(x);sf=torch.empty(size//16,device='cuda')
                    views=tuple(from_dlpack(t) for t in (x,y,sf))
                    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                    f=cute.compile(Fragment(hardware,quant,1),*views,stream)
                    f(*views,stream);torch.cuda.synchronize()
                    if not torch.isfinite(y).all():raise ValueError('nonfinite activation')
                    values['q' if quant else 'activation']=y[:n].reshape(z.shape).cpu().double()
                    if quant:values['scales']=sf[::2][:n//32].reshape(z.shape[0],-1).cpu()
                arms[name]=values
            changed=(arms['hardware']['q']!=arms['exponential']['q'])
            entries=[]
            for r,c in changed.nonzero().tolist()[:128]:
                entries.append(dict(row=int(s['rows'][r]),slot=int(s['slots'][r]),channel=c,z_fp32=float(z[r,c]),activation_fp64=float(own_target[r,c]),activation_exp=float(arms['exponential']['activation'][r,c]),activation_hw=float(arms['hardware']['activation'][r,c]),q_fp64=float(own_q[r,c]),q_exp=float(arms['exponential']['q'][r,c]),q_hw=float(arms['hardware']['q'][r,c])))
            metrics={}
            # Same FP64 FC2 operation, changing only input quantized activation.
            down=s['down'].double().cuda();refout=ideal.cuda()@down.T
            for name,v in arms.items():
                fc2=v['q'].cuda()@down.T
                metrics[name]=dict(activation_vs_same_input_fp64_l2=rel(v['activation'],own_target),activation_max_abs=float((v['activation']-own_target).abs().max()),quantized_vs_same_input_fp64_l2=rel(v['q'],own_q),quantized_vs_fc1_fp64_l2=rel(v['q'],ideal),quantized_changed_elements=int((v['q']!=own_q).sum()),scale_changes=int((v['scales']!=own_sf).sum()),fc2_vs_fp64_pipeline_l2=rel(fc2,refout))
            record['sources'].append(dict(name=source_name,metrics=metrics,changed_between_arms=int(changed.sum()),boundary_examples=entries))
        differences=(actual!=ideal).nonzero()
        examples=[]
        for r,c in differences.tolist()[:256]:
            block=c//32
            scale=float(ideal_scale[r,block])
            examples.append(dict(pool_row=r,token_row=int(s['rows'][r]),slot=int(s['slots'][r]),
                channel=c,z_fp64=float(z64[r,c]),z_fp32=float(s['z_reference'][r,c]),
                activation_fp64=float(target[r,c]),scale_fp64=scale,
                scale_actual=float(decode_e8m0_exact(s['actual_scale_bytes'])[r,block]),
                actual_dequant=float(actual[r,c]),ideal_dequant=float(ideal[r,c])))
        record['actual_vs_ideal_changed_elements']=len(differences)
        record['actual_vs_ideal_examples']=examples
        record['actual_vs_ideal_max_abs']=float((actual-ideal).abs().max())
        reference_fc2=ideal.cuda()@s['down'].double().cuda().T
        actual_fc2=actual.cuda()@s['down'].double().cuda().T
        record['saved_activation_fc2_vs_fp64_pipeline_l2']=rel(actual_fc2,reference_fc2)
        record['saved_kernel_vs_fp64_pipeline_quantized_l2']=rel(actual,ideal)
        record['note']='Independent FC1 operands; actual kernel accumulators unavailable. Ideal FP64 quantizer excludes FP32 reciprocal/multiply rounding. Numerical evidence only, no changed acceptance gates.'
        results.append(record);print(json.dumps(dict(snapshot=str(path),stage='fp64_audit_complete')),flush=True)
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

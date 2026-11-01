import json
import os
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.inkling import language as lang


def separate(kc, vc, norm, k, v, cache, mask=None):
    k, v = kc(k, cache=cache, mask=mask), vc(v, cache=cache, mask=mask)
    return norm(k.reshape(1,1,-1,128)).reshape(k.shape),v


def paired(kc,vc,norm,k,v,cache,mask=None):
    k,v=lang._sconv_kv(kc,vc,k,v,cache,mask)
    return norm(k.reshape(1,1,-1,128)).reshape(k.shape),v


def main():
    mx.random.seed(317)
    C = 1024
    kc = lang.InklingShortConvolution(C, 4, 0)
    vc = lang.InklingShortConvolution(C, 4, 1)
    for c in (kc, vc):
        c.eval()
        c.set_dtype(mx.bfloat16)
    norm=nn.RMSNorm(128,eps=1e-5)
    norm.eval()
    norm.set_dtype(mx.bfloat16)
    projections = [mx.random.normal((1,1,6656)).astype(mx.bfloat16) for _ in range(17)]
    inputs = [(p[...,4096:5120], p[...,5120:6144]) for p in projections]
    states = [mx.random.normal((1,3,C)) for _ in range(2)]
    mx.eval(kc.parameters(),vc.parameters(),projections,states)
    methods = {'paired':paired,'fused':lang._sconv_kv_norm}
    def trial(method, n=2048):
        cache = [mx.array(s) for s in states] + [None, None]
        mx.eval(cache[:2])
        outputs=[]
        begin=time.perf_counter_ns()
        for i in range(n):
            k,v=inputs[i%len(inputs)]
            outputs.extend(method(kc,vc,norm,k,v,cache))
        built=time.perf_counter_ns()
        mx.eval(outputs, cache[:2])
        end=time.perf_counter_ns()
        return {'wall_us':(end-begin)/n/1000,'build_us':(built-begin)/n/1000,'eval_us':(end-built)/n/1000}
    for method in methods.values():
        for _ in range(3):
            trial(method,64)
    rows=[]
    for rep in range(8):
        for tag in ('paired','fused','fused','paired'):
            row={'rep':rep,'tag':tag,'load':os.getloadavg(),**trial(methods[tag])}
            rows.append(row)
            print(json.dumps(row),flush=True)
    summary={tag:{metric:statistics.median(r[metric] for r in rows if r['tag']==tag) for metric in ('wall_us','build_us','eval_us')} for tag in methods}
    summary['wall_speedup']=summary['paired']['wall_us']/summary['fused']['wall_us']
    summary['incremental_fused_speedup']=summary['paired']['wall_us']/summary['fused']['wall_us']
    summary['eval_speedup']=summary['paired']['eval_us']/summary['fused']['eval_us']
    result={'shape':{'B':1,'L':1,'C':C,'projection_width':6656,'dtype':'bfloat16','chain':2048},'summary':summary,'rows':rows}
    Path('convnorm-holdout.json').write_text(json.dumps(result,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()

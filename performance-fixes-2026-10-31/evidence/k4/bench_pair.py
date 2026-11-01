import json
import os
import statistics
import time
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.inkling import language as lang


def separate(kc, vc, k, v, cache, mask=None):
    return kc(k, cache=cache, mask=mask), vc(v, cache=cache, mask=mask)


def main():
    mx.random.seed(317)
    C = 1024
    kc = lang.InklingShortConvolution(C, 4, 0)
    vc = lang.InklingShortConvolution(C, 4, 1)
    for c in (kc, vc):
        c.eval()
        c.set_dtype(mx.bfloat16)
    projections = [mx.random.normal((1,1,6656)).astype(mx.bfloat16) for _ in range(17)]
    inputs = [(p[...,4096:5120], p[...,5120:6144]) for p in projections]
    states = [mx.random.normal((1,3,C)) for _ in range(2)]
    mx.eval(kc.parameters(),vc.parameters(),projections,states)
    methods = {'separate':separate,'paired':lang._sconv_kv}
    def trial(method, n=256):
        cache = [mx.array(s) for s in states] + [None, None]
        mx.eval(cache[:2])
        outputs=[]
        begin=time.perf_counter_ns()
        for i in range(n):
            k,v=inputs[i%len(inputs)]
            outputs.extend(method(kc,vc,k,v,cache))
        built=time.perf_counter_ns()
        mx.eval(outputs, cache[:2])
        end=time.perf_counter_ns()
        return {'wall_us':(end-begin)/n/1000,'build_us':(built-begin)/n/1000,'eval_us':(end-built)/n/1000}
    for method in methods.values():
        for _ in range(3):
            trial(method,64)
    rows=[]
    for rep in range(4):
        for tag in ('separate','paired','paired','separate'):
            row={'rep':rep,'tag':tag,'load':os.getloadavg(),**trial(methods[tag])}
            rows.append(row)
            print(json.dumps(row),flush=True)
    summary={tag:{metric:statistics.median(r[metric] for r in rows if r['tag']==tag) for metric in ('wall_us','build_us','eval_us')} for tag in methods}
    summary['wall_speedup']=summary['separate']['wall_us']/summary['paired']['wall_us']
    summary['eval_speedup']=summary['separate']['eval_us']/summary['paired']['eval_us']
    result={'shape':{'B':1,'L':1,'C':C,'projection_width':6656,'dtype':'bfloat16','chain':256},'summary':summary,'rows':rows}
    Path('paired-conv-benchmark.json').write_text(json.dumps(result,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()

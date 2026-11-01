import json,statistics,time,sys
from pathlib import Path
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).with_name('checkout')))
from mlx_lm.models.inkling_mm_model import ShortConv
from mlx_lm.models.cache import ArraysCache
from combine_prototype import combine
mx.random.seed(83)
c,e,k=4096,8,4
y=mx.random.normal((1,e,c)).astype(mx.bfloat16)*.01
weights=mx.random.uniform(.1,2,(1,e))
w=(mx.random.normal((c,k,1))*.01).astype(mx.bfloat16)
state=mx.random.normal((1,k-1,c)).astype(mx.bfloat16)
res=mx.random.normal((1,1,c)).astype(mx.bfloat16)
conv=ShortConv(c,k,0);conv.weight=w;conv.eval()
mx.eval(y,weights,w,state,res)
def run(fused,n=32):
    cache=ArraysCache(size=1);cache[0]=state
    h=res
    for j in range(n):
        yy=y+h*.01
        if fused:
            h,cache[0]=combine(yy,weights,w,cache[0],h)
        else:
            z=(yy*weights[...,None].astype(yy.dtype)).sum(axis=-2).reshape(1,1,c)
            h=conv(z,cache,residual=h)
    return h,cache[0]
a=run(False);b=run(True);mx.eval(a,b)
print('output exact',mx.array_equal(a[0],b[0]).item(),'state exact',mx.array_equal(a[1],b[1]).item(),'max error',mx.abs(a[0].astype(mx.float32)-b[0].astype(mx.float32)).max().item(),flush=True)
for _ in range(4):mx.eval(run(False));mx.eval(run(True))
times={'base':[],'fused':[]}
for j in range(16):
    order=('base','fused') if j%2==0 else ('fused','base')
    for label in order:
        t=time.perf_counter();mx.eval(run(label=='fused'));times[label].append((time.perf_counter()-t)*1e6/32)
r={'base_us':statistics.median(times['base']),'fused_us':statistics.median(times['fused']),'samples':times}
r['speedup']=r['base_us']/r['fused_us'];print(json.dumps(r),flush=True)
Path(__file__).with_name('combine-results.json').write_text(json.dumps(r,indent=2))

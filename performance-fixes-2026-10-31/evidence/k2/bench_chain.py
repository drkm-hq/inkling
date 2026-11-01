import json, statistics, time
from pathlib import Path
import mlx.core as mx
from prototype import make_kernel
import sys
sys.path.insert(0,str(Path(__file__).with_name('checkout')))
from mlx_lm.models.activations import swiglu
mx.random.seed(43)
N,E,D,I,K,G=1,32,4096,2048,8,64

def packed(out,in_dim):
    w=mx.random.randint(0,2**32-1,(E,out,in_dim//16),dtype=mx.uint32)
    s=mx.random.uniform(.004,.012,(E,out,in_dim//G)).astype(mx.bfloat16)
    b=(-1.5*s).astype(mx.bfloat16)
    mx.eval(w,s,b)
    return w,s,b
up,gate,down=packed(I,D),packed(I,D),packed(D,I)
x=mx.random.normal((N,D)).astype(mx.bfloat16)
indices=[mx.array([[(j*7+e*3)%E for e in range(6)]+[30,31]],dtype=mx.uint32) for j in range(32)]
w=mx.array([[1.1,.9,.8,.7,.6,.5,.4,.3]],dtype=mx.bfloat16)
mx.eval(x,indices,w)
def qmm(x,p,inds):return mx.gather_qmm(x,*p,rhs_indices=inds,transpose=True,group_size=G,bits=2)
def baseline(h,inds):return swiglu(qmm(h[:,None,None],gate,inds),qmm(h[:,None,None],up,inds)).squeeze(-2)
def chain(fun,n=32):
    h=x
    for j in range(n):
        inds=indices[j%len(indices)]
        a=fun(h,inds)
        y=qmm(a[:,:,None],down,inds).squeeze(-2)
        z=(y*w[:,:,None]).sum(axis=-2)
        h=h+z*.01
    return h
ref=chain(baseline);mx.eval(ref)
results=[]
for rows,sg in ((2,1),(2,2),(4,1),(4,2),(4,4),(2,4),(8,1)):
    kernel=make_kernel(rows,sg,activation=2)
    fun=lambda h,inds:kernel(h,up,gate,inds)
    test=chain(fun);mx.eval(test)
    print('correctness',rows,sg,'exact',mx.array_equal(test,ref).item(),'max',mx.abs(test.astype(mx.float32)-ref.astype(mx.float32)).max().item(),flush=True)
    for _ in range(2):mx.eval(chain(fun));mx.eval(chain(baseline))
    times={'base':[],'fused':[]}
    for repeat in range(10):
        order=('base','fused') if repeat%2==0 else ('fused','base')
        for label in order:
            f=baseline if label=='base' else fun
            t=time.perf_counter();mx.eval(chain(f));elapsed=time.perf_counter()-t
            times[label].append(elapsed*1e6/32)
    result={'rows':rows,'sg':sg,'chain_steps':32,'base_us':statistics.median(times['base']),'fused_us':statistics.median(times['fused']),'samples':times}
    result['speedup']=result['base_us']/result['fused_us']
    results.append(result);print(json.dumps(result),flush=True)
Path(__file__).with_name('chain-results.json').write_text(json.dumps(results,indent=2))

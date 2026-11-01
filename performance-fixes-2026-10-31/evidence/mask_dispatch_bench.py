import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.inkling import language as lang

mx.random.seed(92)
rows = []
for L, S, E, window in [(1,512,512,512),(1,32768,1024,0),(128,2048,512,512),(2048,8192,1024,0)]:
    packed = mx.random.normal((1,L,6656)).astype(mx.bfloat16)
    rel = packed[..., -512:].reshape(1,L,32,16)
    proj = mx.random.normal((16,E)).astype(mx.bfloat16)
    shape = mx.zeros((1,8,S,128), dtype=mx.bfloat16)
    mx.eval(rel,proj,shape)
    groups = [(8,8),(32,1),(64,1),(128,1),(32,8),(64,4)]
    if L>1:
        groups = [(8,8),(32,8),(64,4),(128,2)]
    def call(r, group):
        gx,gy = group
        return lang._mask_v2_kernel(inputs=[r,proj,shape],
            template=[('T',mx.bfloat16),('D_REL',16),('REL_EXTENT',E),('SLIDING',window)],
            grid=(lang._rup(S,gx),lang._rup(L,gy),32),threadgroup=(gx,gy,1),
            output_shapes=[(1,32,L,S)],output_dtypes=[mx.bfloat16])[0]
    reference=call(rel,groups[0]); mx.eval(reference)
    for group in groups:
        candidate=call(rel,group)
        assert mx.array_equal(reference,candidate).item(), (L,S,group)
    del candidate, reference
    mx.clear_cache()
    count=128 if L==1 else 1
    samples={str(g):[] for g in groups}
    for repeat in range(8):
        order=groups if repeat%2==0 else list(reversed(groups))
        for group in order:
            mx.synchronize()
            start=time.perf_counter()
            outs=[call(rel+mx.array((i+1)*0.00001,rel.dtype),group) for i in range(count)]
            mx.eval(outs)
            elapsed=(time.perf_counter()-start)*1e6/count
            samples[str(group)].append(elapsed)
            del outs
    row={'L':L,'S':S,'extent':E,'window':window,'count':count,'exact':True,
         'median_us':{g:statistics.median(v) for g,v in samples.items()},'samples_us':samples}
    rows.append(row)
    print(json.dumps(row),flush=True)
    del packed,rel,proj,shape
    mx.clear_cache()
Path('work/perf/mask-dispatch-results.json').write_text(json.dumps(rows,indent=2))

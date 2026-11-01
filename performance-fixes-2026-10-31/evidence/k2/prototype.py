import mlx.core as mx

SOURCE = r'''
    uint lane = thread_position_in_threadgroup.x % 32;
    uint simd = thread_position_in_threadgroup.x / 32;
    uint selected = threadgroup_position_in_grid.y;
    uint expert = inds[selected];
    uint n = selected / K;
    uint first = threadgroup_position_in_grid.x * (SG * ROWS) + simd * ROWS;
    float a[ROWS] = {0};
    float b[ROWS] = {0};
    const device uchar *uw = (const device uchar *)upw;
    const device uchar *gw = (const device uchar *)gatew;
    for (uint start = 0; start < D; start += 512) {
        float v[16];
        float sum = 0;
        uint xi = n * D + start + lane * 16;
        for (uint j = 0; j < 16; j += 4) {
            sum += float(x[xi+j]) + float(x[xi+j+1]) + float(x[xi+j+2]) + float(x[xi+j+3]);
            v[j] = float(x[xi+j]);
            v[j+1] = float(x[xi+j+1]) / 4.0f;
            v[j+2] = float(x[xi+j+2]) / 16.0f;
            v[j+3] = float(x[xi+j+3]) / 64.0f;
        }
        for (uint r = 0; r < ROWS; ++r) {
            uint row = expert * I + first + r;
            uint wi = row * (D/4) + start/4 + lane*4;
            uint si = row * (D/G) + (start + lane*16)/G;
            float au = 0;
            float ag = 0;
            for (uint j = 0; j < 4; ++j) {
                uchar u = uw[wi+j];
                uchar g = gw[wi+j];
                au += v[4*j]*(u&3) + v[4*j+1]*(u&12) + v[4*j+2]*(u&48) + v[4*j+3]*(u&192);
                ag += v[4*j]*(g&3) + v[4*j+1]*(g&12) + v[4*j+2]*(g&48) + v[4*j+3]*(g&192);
            }
            a[r] += float(ups[si])*au + sum*float(upb[si]);
            b[r] += float(gates[si])*ag + sum*float(gateb[si]);
        }
    }
    for (uint r = 0; r < ROWS; ++r) {
        float uu = simd_sum(a[r]);
        float gg = simd_sum(b[r]);
        if (lane == 0) {
            T u = T(uu);
            T g = T(gg);
            STORE
        }
    }
'''

def make_kernel(rows=4, sg=2, fused=True, activation=0):
    store = '''float z = 1.0f / (1.0f + metal::precise::exp(metal::abs(float(g))));
            T s = T(g < T(0) ? z : 1.0f-z);
            out[selected*I+first+r] = T(T(g*s)*u);'''
    if activation == 1:
        store = '''float z = 1.0f / (1.0f + metal::precise::exp(metal::abs(float(g))));
            float s = float(g) < 0 ? z : 1.0f-z;
            out[selected*I+first+r] = T(float(g)*s*float(u));'''
    if activation == 2:
        store = '''auto z = 1 / (1 + metal::precise::exp(metal::abs(g)));
            T s = g < T(0) ? z : 1-z;
            out[selected*I+first+r] = T(T(g*s)*u);'''
    if not fused:
        store = '''out[selected*(2*I)+first+r] = u;
            out[selected*(2*I)+I+first+r] = g;'''
    kernel=mx.fast.metal_kernel(name=f'inkling_gateup_q2_r{rows}_s{sg}_f{int(fused)}_a{activation}',input_names=['x','upw','ups','upb','gatew','gates','gateb','inds'], output_names=['out'],source=SOURCE.replace('STORE',store))
    def call(x,up,gate,inds):
        n,d=x.shape
        _,i,_=up[0].shape
        k=inds.shape[-1]
        return kernel(inputs=[x,*up,*gate,inds],template=[('T',x.dtype),('D',d),('I',i),('K',k),('G',d//up[1].shape[-1]),('ROWS',rows),('SG',sg)],grid=((i//(rows*sg))*sg*32,n*k,1),threadgroup=(sg*32,1,1),output_shapes=[(n,k,i if fused else 2*i)],output_dtypes=[x.dtype])[0]
    return call

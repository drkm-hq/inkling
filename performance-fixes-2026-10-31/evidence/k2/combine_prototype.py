import mlx.core as mx

SOURCE = r'''
    uint c = thread_position_in_grid.x;
    T sum = T(0);
    for (uint e = 0; e < E; ++e) {
        T product = T(y[e*C+c] * T(weights[e]));
        sum = T(product + sum);
    }
    T combined = T(sum);
    float result = float(combined);
    for (uint k = 0; k < K; ++k) {
        float input = k == K-1 ? float(combined) : float(state[k*C+c]);
        result += input * float(w[c*K+k]);
    }
    T convolved = T(result);
    out[c] = T(float(convolved) + float(residual[c]));
    for (uint k = 0; k < K-1; ++k) {
        new_state[k*C+c] = k == K-2 ? combined : state[(k+1)*C+c];
    }
'''
kernel=mx.fast.metal_kernel(name='inkling_combine_sconv',input_names=['y','weights','w','state','residual'],output_names=['out','new_state'],source=SOURCE)
def combine(y,weights,w,state,residual):
    _,e,c=y.shape
    k=w.shape[1]
    return kernel(inputs=[y,weights,w,state,residual],template=[('T',y.dtype),('E',e),('C',c),('K',k)],grid=(c,1,1),threadgroup=(min(c,256),1,1),output_shapes=[(1,1,c),(1,k-1,c)],output_dtypes=[y.dtype,y.dtype])

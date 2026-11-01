#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstdint>
#include <cstring>
using Clock=std::chrono::steady_clock;
int main(int argc,char **argv) {
 int layers=argc>1?atoi(argv[1]):168, channels=argc>2?atoi(argv[2]):0, taps=argc>3?atoi(argv[3]):4;
 int tokens=argc>4?atoi(argv[4]):1, seq=argc>5?atoi(argv[5]):1, reps=argc>6?atoi(argv[6]):100;
 auto backend=ggml_backend_metal_init(); if(!backend)return 2;
 ggml_init_params params{16*1024*1024,nullptr,true}; auto ctx=ggml_init(params);
 auto graph=ggml_new_graph_custom(ctx,8192,false);
 std::vector<ggml_tensor*> inputs,weights,outputs;
 for(int l=0;l<layers;++l) {
  int c=channels?channels:(l%4==3?4096:1024);
  auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,taps+tokens-1,c,seq);
  auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,taps,c);
  auto y=ggml_ssm_conv(ctx,x,w); ggml_set_output(y);
  inputs.push_back(x);weights.push_back(w);outputs.push_back(y); ggml_build_forward_expand(graph,y);
 }
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 3;
 std::mt19937 rng(1234);std::uniform_real_distribution<float> dist(-.5f,.5f);
 std::vector<std::vector<float>> in,wt;
 for(int l=0;l<layers;++l) {
  in.emplace_back(ggml_nelements(inputs[l])); wt.emplace_back(ggml_nelements(weights[l]));
  for(auto&v:in.back())v=dist(rng);for(auto&v:wt.back())v=dist(rng);
  ggml_backend_tensor_set(inputs[l],in.back().data(),0,ggml_nbytes(inputs[l]));
  ggml_backend_tensor_set(weights[l],wt.back().data(),0,ggml_nbytes(weights[l]));
 }
 auto compute=[&]{if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)std::abort();};
 compute();uint64_t hash=1469598103934665603ull;double maxerr=0;
 for(int l=0;l<layers;++l) {
  auto y=outputs[l]; int c=y->ne[0];std::vector<float> result(ggml_nelements(y));
  ggml_backend_tensor_get(y,result.data(),0,ggml_nbytes(y));
  for(int s=0;s<seq;++s)for(int t=0;t<tokens;++t)for(int ch=0;ch<c;++ch) {
   double expected=0;for(int k=0;k<taps;++k) expected+=double(in[l][(s*c+ch)*(taps+tokens-1)+t+k])*wt[l][ch*taps+k];
   float actual=result[(s*tokens+t)*c+ch]; if(!std::isfinite(actual))return 4;
   maxerr=std::max(maxerr,std::abs(expected-actual));
   uint32_t bits;memcpy(&bits,&actual,4);hash=(hash^bits)*1099511628211ull;
  }
 }
 printf("CHECK layers=%d channels=%d taps=%d tokens=%d seq=%d hash=%016llx max_abs=%.9g\n",layers,channels,taps,tokens,seq,(unsigned long long)hash,maxerr);
 if(maxerr>2e-6)return 5;
 for(int i=0;i<8;++i)compute();
 for(int r=0;r<3;++r){auto start=Clock::now();for(int i=0;i<reps;++i)compute();double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/reps;printf("TIME round=%d us=%.6f\n",r,us);}
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}

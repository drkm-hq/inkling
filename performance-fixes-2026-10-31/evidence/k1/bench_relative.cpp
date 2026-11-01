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

using Clock = std::chrono::steady_clock;
using FusionGet = void * (*)(ggml_backend_dev_t);
using FusionEnable = void (*)(void *, bool);
using FusionInit = void (*)(void *);
using FusionStats = int (*)(void *, const char **, uint64_t *, int);

int main(int argc, char ** argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 42;
    int columns = argc > 2 ? atoi(argv[2]) : 32;
    int extent = argc > 3 ? atoi(argv[3]) : 0;
    int retain = argc > 4 ? atoi(argv[4]) : 0;
    auto backend = ggml_backend_metal_init();
    if (!backend) return 2;
    auto dev = ggml_backend_get_device(backend);
    auto reg = ggml_backend_dev_backend_reg(dev);
    auto get = (FusionGet)ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get");
    auto enable = (FusionEnable)ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_set_enabled");
    auto init = (FusionInit)ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init");
    auto stats = (FusionStats)ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_get");
    auto reset = (FusionInit)ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_reset");
    if (!get || !enable || !stats || !init || !reset) return 3;
    void * finfo = get(dev);
    init(finfo);
    ggml_init_params params{16*1024*1024, nullptr, true};
    auto ctx = ggml_init(params);
    auto graph = ggml_new_graph_custom(ctx, 8192, false);
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, columns);
    ggml_set_input(input);
    ggml_tensor * x = input;
    std::vector<ggml_tensor *> weights, outputs, copies;
    for (int l = 0; l < layers; ++l) {
        int e = extent ? extent : (l % 6 == 0 ? 1024 : 512);
        auto weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, e, 16);
        weights.push_back(weight);
        auto cont = ggml_cont(ctx, ggml_transpose(ctx, weight));
        if (retain) ggml_set_output(cont);
        copies.push_back(cont);
        auto y = ggml_mul_mat(ctx, cont, x);
        ggml_prec_set_acc(y, GGML_PREC_F32_PEDANTIC);
        outputs.push_back(y);
        if (l + 1 < layers) {
            x = ggml_add(ctx, x, ggml_view_2d(ctx, y, 16, columns, y->nb[1], 0));
        } else x = y;
    }
    ggml_build_forward_expand(graph, x);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 4;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    for (auto t : weights) {
        std::vector<float> values(ggml_nelements(t));
        for (auto & v : values) v = distribution(rng);
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
    }
    std::vector<float> values(ggml_nelements(input));
    for (auto & v : values) v = distribution(rng);
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    auto compute = [&] {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
    };
    enable(finfo, false); compute();
    std::vector<std::vector<float>> reference;
    for (auto t : outputs) {
        reference.emplace_back(ggml_nelements(t));
        ggml_backend_tensor_get(t, reference.back().data(), 0, ggml_nbytes(t));
    }
    enable(finfo, true); reset(finfo); compute();
    double max_abs = 0, sum_sq = 0, ref_sq = 0;
    for (int i = 0; i < layers; ++i) {
        std::vector<float> result(reference[i].size());
        ggml_backend_tensor_get(outputs[i], result.data(), 0, ggml_nbytes(outputs[i]));
        for (size_t j=0; j<result.size(); ++j) {
            if (!std::isfinite(result[j])) return 5;
            double err = result[j]-reference[i][j];
            max_abs = std::max(max_abs, std::abs(err));
            sum_sq += err*err; ref_sq += double(reference[i][j])*reference[i][j];
        }
    }
    int count = stats(finfo, nullptr, nullptr, 0);
    std::vector<const char *> labels(count); std::vector<uint64_t> counts(count);
    stats(finfo, labels.data(), counts.data(), count);
    uint64_t fused_count = 0;
    for (int i=0;i<count;++i) if (std::string(labels[i]) == "CONT+MUL_MAT") fused_count += counts[i];
    printf("CHECK layers=%d columns=%d extent=%d retained=%d fused=%llu max_abs=%.9g nmse=%.9g\n", layers,columns,extent,retain,(unsigned long long)fused_count,max_abs,sum_sq/ref_sq);
    bool expected_fusion = columns <= 64 && !retain && (!extent || (extent>=32 && extent<=2048));
    if ((expected_fusion ? fused_count < uint64_t(std::max(1, layers-2)) : fused_count != 0) || max_abs>2e-6 || sum_sq/ref_sq>1e-10) return 6;
    // Independent scalar double reference for a single projection.
    if (layers==1) {
        auto w=weights[0]; int e=w->ne[0];
        std::vector<float> data(ggml_nelements(w));
        ggml_backend_tensor_get(w,data.data(),0,ggml_nbytes(w));
        double err=0;
        for(int c=0;c<columns;++c) for(int row=0;row<e;++row) {
            double s=0; for(int k=0;k<16;++k) s+=double(data[k*e+row])*values[c*16+k];
            err=std::max(err,std::abs(s-reference[0][c*e+row]));
        }
        printf("ORACLE max_abs=%.9g\n",err); if(err>2e-6) return 7;
    }
    for(int mode: {0,1}) {enable(finfo,mode);for(int i=0;i<8;++i)compute();}
    for(int round=0;round<9;++round) for(int side=0;side<2;++side) {
        bool fused=(round%2) ? side==0 : side==1;
        enable(finfo,fused);
        auto start=Clock::now();
        int reps=40; for(int i=0;i<reps;++i) compute();
        double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/reps;
        printf("TIME round=%d fused=%d us=%.6f\n",round,fused,us);
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
}

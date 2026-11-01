// sched_bench: cost of the scheduler fallback that Metal pays for GGML_OP_FLASH_ATTN_EXT_BANDED.
// Builds a 42-"layer" toy graph on a {MTL0, CPU} scheduler. Each layer: a Metal elementwise op,
// an attention op, a Metal elementwise op. With variant=banded the attention op is the banded FA
// (unsupported on Metal -> runs on the CPU backend, one graph split before and after each layer,
// exactly like llama.cpp with -fa on). With variant=fa it is ggml_flash_attn_ext (stays on Metal).
// 7 of the 42 layers use n_kv_global (the global layers), the rest n_kv_swa (window 512 + batch).
//
// usage: sched_bench <variant fa|banded> <n_q> <n_kv_global> <n_kv_swa> [--threads N] [--reps N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void fill(ggml_tensor * t, float lo, float hi, uint32_t seed) {
    const int64_t n = ggml_nelements(t);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> f(n);
    for (int64_t i = 0; i < n; ++i) f[i] = dist(rng);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n*sizeof(float));
    } else {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, n*sizeof(ggml_fp16_t));
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <fa|banded> <n_q> <n_kv_global> <n_kv_swa> [--threads N] [--reps N]\n", argv[0]); return 1; }
    const std::string variant = argv[1];
    const int64_t n_q = atoll(argv[2]);
    const int64_t n_kv_g = atoll(argv[3]);
    const int64_t n_kv_s = atoll(argv[4]);
    int n_threads = 12, reps = 20;
    for (int i = 5; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--threads") n_threads = atoi(argv[i+1]);
        else if (k == "--reps") reps = atoi(argv[i+1]);
    }
    const int64_t D = 128, H = 32, Hkv = 8, E = 1024, L = 42;
    bool is_global[42] = {}; for (int l : {5, 11, 17, 23, 29, 35, 41}) is_global[l] = true;

    ggml_backend_load_all();
    ggml_backend_t metal = ggml_backend_init_by_name("MTL0", nullptr);
    ggml_backend_t cpu   = ggml_backend_init_by_name("CPU", nullptr);
    if (!metal || !cpu) { fprintf(stderr, "backends missing\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu, n_threads);

    ggml_backend_t backends[2] = { metal, cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(metal), ggml_backend_get_default_buffer_type(cpu) };

    // inputs on Metal (host-visible unified memory; the CPU backend reads them in place)
    ggml_init_params ip = { (size_t) (8*L + 16)*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_in = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, D*H, n_q);
    std::vector<ggml_tensor *> ks(L), vs(L), ms(L), rs(L);
    for (int l = 0; l < L; ++l) {
        const int64_t n_kv = is_global[l] ? n_kv_g : n_kv_s;
        ks[l] = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, D, n_kv, Hkv, 1);
        vs[l] = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, D, n_kv, Hkv, 1);
        ms[l] = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, n_kv, n_q, 1, 1);
        rs[l] = variant == "banded" ? ggml_new_tensor_4d(ctx_in, GGML_TYPE_F32, E, H, n_q, 1) : nullptr;
    }
    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, metal);
    if (!buf_in) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill(x, -1, 1, 1);
    for (int l = 0; l < L; ++l) {
        const int64_t n_kv = is_global[l] ? n_kv_g : n_kv_s;
        fill(ks[l], -1, 1, 10 + l); fill(vs[l], -1, 1, 100 + l);
        if (rs[l]) fill(rs[l], -0.25f, 0.25f, 200 + l);
        std::vector<float> m((size_t) n_kv*n_q);
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t p = n_kv - n_q + i;
            for (int64_t j = 0; j < n_kv; ++j) {
                const bool vis = j <= p && (is_global[l] || p - j < 512);
                m[i*n_kv + j] = vis ? 0.0f : -INFINITY;
            }
        }
        std::vector<ggml_fp16_t> h(m.size());
        ggml_fp32_to_fp16_row(m.data(), h.data(), m.size());
        ggml_backend_tensor_set(ms[l], h.data(), 0, h.size()*sizeof(ggml_fp16_t));
    }

    ggml_init_params gp = { (size_t) ggml_tensor_overhead()*32*L + ggml_graph_overhead_custom(16*L, false), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16*L, false);

    ggml_tensor * cur = x;
    for (int l = 0; l < L; ++l) {
        ggml_tensor * q = ggml_scale(ctx, cur, 0.5f);                      // Metal op before attention
        ggml_tensor * q4 = ggml_permute(ctx, ggml_reshape_3d(ctx, q, D, H, n_q), 0, 2, 1, 3); // [D, n_q, H]
        ggml_tensor * att;
        if (variant == "banded") {
            att = ggml_flash_attn_ext_banded(ctx, q4, ks[l], vs[l], ms[l], rs[l], 1.0f/D, E);
        } else {
            att = ggml_flash_attn_ext(ctx, q4, ks[l], vs[l], ms[l], 1.0f/D, 0.0f, 0.0f);
        }
        ggml_prec_set_acc(att, GGML_PREC_F32);
        ggml_tensor * o = ggml_reshape_2d(ctx, att, D*H, n_q);
        cur = ggml_add(ctx, cur, o);                                       // Metal op after attention
    }
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, ggml_graph_n_nodes(gf) + 8*L + 64, false, true);
    if (!ggml_backend_sched_reserve(sched, gf)) { fprintf(stderr, "reserve failed\n"); return 1; }

    for (int i = 0; i < 3; ++i) {
        ggml_backend_sched_reset(sched);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        ggml_backend_sched_synchronize(sched);
    }
    const int n_splits = ggml_backend_sched_get_n_splits(sched);
    std::vector<double> us;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_sched_synchronize(sched);
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    double mean = 0; for (double u : us) mean += u; mean /= us.size();
    printf("RESULT sched variant=%s n_q=%lld n_kv_global=%lld n_kv_swa=%lld threads=%d nodes=%d splits=%d us_mean=%.1f us_median=%.1f us_min=%.1f\n",
           variant.c_str(), (long long) n_q, (long long) n_kv_g, (long long) n_kv_s, n_threads, ggml_graph_n_nodes(gf), n_splits,
           mean, us[us.size()/2], us[0]);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);
    ggml_backend_free(metal);
    ggml_backend_free(cpu);
    return 0;
}

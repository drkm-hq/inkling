// fa_bench: time the attention variants that llama.cpp PR #25731 builds for Inkling, at Inkling-Small
// shapes, on one ggml backend (MTL0 or CPU). The tensor layouts (KV cache strides, mask types, the
// rel-bias gather chain) replicate src/models/inkling.cpp and src/llama-graph.cpp build_attn_mha.
//
// usage: fa_bench <backend> <variant> <n_q> <n_kv> [--swa N] [--extent E] [--threads N] [--reps N]
//   backend : MTL0 | CPU
//   variant : fa        ggml_flash_attn_ext (f16 KV, f16 mask, no bias)   -> the existing Metal FA, upper bound
//             banded    ggml_flash_attn_ext_banded (f16 KV, f16 mask, f32 rel) -> the op Metal lacks (CPU only)
//             dense     the -fa off path: scale q, mul_mat(k,q), rel pad/permute/get_rows/permute, add,
//                       soft_max_ext(f32 mask), mul_mat(v^T, kq), permute+cont
//             dense-nobias  same without the rel-bias chain
//             bias      only the rel-bias materialization chain (pad, cont, get_rows, cont)
//             qk        only mul_mat(k, q)
// output: one line "RESULT backend=... variant=... n_q=... n_kv=... swa=... us_mean=... us_min=... gflops=..."

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

struct cfg_t {
    std::string backend = "MTL0";
    std::string variant = "fa";
    int64_t n_q   = 512;
    int64_t n_kv  = 8192;
    int64_t swa   = 0;      // 0 = causal only, >0 = sliding window (STANDARD: visible iff pos_q - pos_k < swa)
    int64_t E     = 1024;   // rel extent
    int64_t H     = 32;
    int64_t Hkv   = 8;
    int64_t D     = 128;
    int64_t kv_size = 0;    // allocated cache cells (>= n_kv); 0 = n_kv rounded up to 256
    int     n_threads = 12;
    int     reps  = 10;
    bool    pool  = false;  // persistent CPU threadpool (as llama-cli/server attach), avoids per-graph thread spawn
};

static void fill_rand(ggml_tensor * t, float lo, float hi, uint32_t seed) {
    const int64_t n = ggml_nelements(t);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> f(n);
    for (int64_t i = 0; i < n; ++i) f[i] = dist(rng);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n*sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, n*sizeof(ggml_fp16_t));
    } else {
        fprintf(stderr, "fill_rand: unsupported type\n");
        exit(1);
    }
}

int main(int argc, char ** argv) {
    cfg_t c;
    if (argc < 5) {
        fprintf(stderr, "usage: %s <backend> <variant> <n_q> <n_kv> [--swa N] [--extent E] [--threads N] [--reps N] [--kv-size N]\n", argv[0]);
        return 1;
    }
    c.backend = argv[1];
    c.variant = argv[2];
    c.n_q  = atoll(argv[3]);
    c.n_kv = atoll(argv[4]);
    for (int i = 5; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if      (k == "--swa")     c.swa = atoll(argv[i+1]);
        else if (k == "--extent")  c.E = atoll(argv[i+1]);
        else if (k == "--threads") c.n_threads = atoi(argv[i+1]);
        else if (k == "--reps")    c.reps = atoi(argv[i+1]);
        else if (k == "--kv-size") c.kv_size = atoll(argv[i+1]);
        else if (k == "--pool")    { c.pool = atoi(argv[i+1]) != 0; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (c.kv_size == 0) c.kv_size = ((c.n_kv + 255)/256)*256;
    if (c.kv_size < c.n_kv) { fprintf(stderr, "kv_size < n_kv\n"); return 1; }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(c.backend.c_str(), nullptr);
    if (!backend) { fprintf(stderr, "backend %s not found\n", c.backend.c_str()); return 1; }
    ggml_threadpool_t tp = nullptr;
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, c.n_threads);
        if (c.pool) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(c.n_threads);
            tp = ggml_threadpool_new(&tpp);
            ggml_backend_cpu_set_threadpool(backend, tp);
        }
    }

    const int64_t D = c.D, H = c.H, Hkv = c.Hkv, n_q = c.n_q, n_kv = c.n_kv, E = c.E, kv_size = c.kv_size;
    const bool fa_like   = c.variant == "fa" || c.variant == "banded";
    const bool need_rel  = c.variant == "banded" || c.variant == "dense" || c.variant == "bias";
    const bool need_idx  = c.variant == "dense" || c.variant == "bias";
    const bool need_mask = c.variant != "bias" && c.variant != "qk";
    const bool v_trans   = !fa_like; // llama.cpp: V cache is transposed iff flash_attn is off

    // ---- inputs (allocated on the backend)
    ggml_init_params ip = { 32*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_in = ggml_init(ip);

    ggml_tensor * q  = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F32, D, H, n_q, 1);        // [D, H, n_tokens] as produced by the model
    ggml_tensor * kc = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, D*Hkv, kv_size);      // K cache storage [n_embd_k_gqa, kv_size]
    ggml_tensor * vc = v_trans ? ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, kv_size, D*Hkv)   // V cache transposed [kv_size, n_embd_v_gqa]
                               : ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, D*Hkv, kv_size);
    ggml_tensor * mask = nullptr;
    if (need_mask) {
        mask = ggml_new_tensor_4d(ctx_in, fa_like ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_q, 1, 1);
    }
    ggml_tensor * rel = need_rel ? ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, E, H, n_q) : nullptr; // mul_mat(proj, r) output
    ggml_tensor * idx = need_idx ? ggml_new_tensor_1d(ctx_in, GGML_TYPE_I32, n_kv*n_q) : nullptr;

    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    if (!buf_in) { fprintf(stderr, "input alloc failed\n"); return 1; }

    fill_rand(q,  -1.0f, 1.0f, 1);
    fill_rand(kc, -1.0f, 1.0f, 2);
    fill_rand(vc, -1.0f, 1.0f, 3);
    if (rel) fill_rand(rel, -0.25f, 0.25f, 4);

    // mask: row i is query pos p_i = n_kv - n_q + i (the banded alignment); key j visible iff j <= p_i and (swa == 0 || p_i - j < swa)
    if (mask) {
        std::vector<float> m((size_t) n_kv*n_q);
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t p = n_kv - n_q + i;
            for (int64_t j = 0; j < n_kv; ++j) {
                const bool vis = j <= p && (c.swa == 0 || p - j < c.swa);
                m[i*n_kv + j] = vis ? 0.0f : -INFINITY;
            }
        }
        if (mask->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(mask, m.data(), 0, m.size()*sizeof(float));
        } else {
            std::vector<ggml_fp16_t> h(m.size());
            ggml_fp32_to_fp16_row(m.data(), h.data(), m.size());
            ggml_backend_tensor_set(mask, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
        }
    }
    // idx: set_input_pos_rel_flat semantics, idx(i, j) = i*(E+1) + rel, rel = E when out of band
    if (idx) {
        std::vector<int32_t> ix((size_t) n_kv*n_q);
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t p = n_kv - n_q + i;
            for (int64_t j = 0; j < n_kv; ++j) {
                const int64_t d = p - j;
                const int32_t r = (d >= 0 && d < E) ? (int32_t) d : (int32_t) E;
                ix[i*n_kv + j] = (int32_t) (i*(E + 1)) + r;
            }
        }
        ggml_backend_tensor_set(idx, ix.data(), 0, ix.size()*sizeof(int32_t));
    }

    // ---- graph
    ggml_init_params gp = { ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // KV views exactly as llama_kv_cache::get_k / get_v (ns = 1) followed by the model's permute(0,2,1,3)
    ggml_tensor * k = ggml_view_4d(ctx, kc, D, Hkv, n_kv, 1,
            ggml_row_size(GGML_TYPE_F16, D), ggml_row_size(GGML_TYPE_F16, D*Hkv), ggml_row_size(GGML_TYPE_F16, D*Hkv*kv_size), 0);
    ggml_tensor * v;
    if (!v_trans) {
        v = ggml_view_4d(ctx, vc, D, Hkv, n_kv, 1,
            ggml_row_size(GGML_TYPE_F16, D), ggml_row_size(GGML_TYPE_F16, D*Hkv), ggml_row_size(GGML_TYPE_F16, D*Hkv*kv_size), 0);
    } else {
        v = ggml_view_4d(ctx, vc, n_kv, Hkv, D, 1,
            ggml_row_size(GGML_TYPE_F16, kv_size*D), ggml_row_size(GGML_TYPE_F16, kv_size), ggml_row_size(GGML_TYPE_F16, kv_size*D*Hkv), 0);
    }
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3); // [D, n_q, H]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);                // [D, n_kv, Hkv]
    v = ggml_permute(ctx, v, 0, 2, 1, 3);                // [D, n_kv, Hkv] or [n_kv, D, Hkv] (transposed)

    const float scale = 1.0f/float(D);
    ggml_tensor * out = nullptr;
    double flops = 4.0*H*n_q*n_kv*D; // QK + PV, 2 flop per MAC

    if (c.variant == "fa") {
        out = ggml_flash_attn_ext(ctx, qp, k, v, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
    } else if (c.variant == "banded") {
        ggml_tensor * rel_fa = ggml_reshape_4d(ctx, rel, E, H, n_q, 1);
        out = ggml_flash_attn_ext_banded(ctx, qp, k, v, mask, rel_fa, scale, E);
        ggml_prec_set_acc(out, GGML_PREC_F32);
    } else if (c.variant == "qk") {
        out = ggml_mul_mat(ctx, k, ggml_scale(ctx, qp, scale));
        ggml_prec_set_acc(out, GGML_PREC_F32);
        flops = 2.0*H*n_q*n_kv*D;
    } else if (c.variant == "bias") {
        ggml_tensor * r = ggml_pad(ctx, rel, 1, 0, 0, 0);
        r = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));
        r = ggml_reshape_2d(ctx, r, H, (E + 1)*n_q);
        ggml_tensor * kq_b = ggml_get_rows(ctx, r, idx);
        kq_b = ggml_reshape_3d(ctx, kq_b, H, n_kv, n_q);
        out = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3)); // [n_kv, n_q, H]
        flops = 0.0;
    } else if (c.variant == "dense" || c.variant == "dense-nobias") {
        ggml_tensor * qs = ggml_scale(ctx, qp, scale);
        ggml_tensor * kq = ggml_mul_mat(ctx, k, qs); // [n_kv, n_q, H]
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        if (c.variant == "dense") {
            ggml_tensor * r = ggml_pad(ctx, rel, 1, 0, 0, 0);
            r = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));
            r = ggml_reshape_2d(ctx, r, H, (E + 1)*n_q);
            ggml_tensor * kq_b = ggml_get_rows(ctx, r, idx);
            kq_b = ggml_reshape_3d(ctx, kq_b, H, n_kv, n_q);
            kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));
            kq = ggml_add(ctx, kq, kq_b);
        }
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq); // [D, n_q, H]
        out = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), D*H, n_q);
    } else {
        fprintf(stderr, "unknown variant %s\n", c.variant.c_str());
        return 1;
    }
    ggml_build_forward_expand(gf, out);

    // every node must be supported by the backend (no sched fallback here)
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, n)) {
            printf("RESULT backend=%s variant=%s n_q=%lld n_kv=%lld swa=%lld status=UNSUPPORTED op=%s\n",
                   c.backend.c_str(), c.variant.c_str(), (long long) n_q, (long long) n_kv, (long long) c.swa, ggml_op_desc(n));
            return 0;
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "graph alloc failed\n"); return 1; }
    const size_t compute_bytes = ggml_gallocr_get_buffer_size(galloc, 0);

    // warmup
    for (int i = 0; i < 2; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        ggml_backend_synchronize(backend);
    }

    std::vector<double> us;
    for (int i = 0; i < c.reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double mean = 0; for (double x : us) mean += x; mean /= us.size();
    double mn = *std::min_element(us.begin(), us.end());

    printf("RESULT backend=%s variant=%s n_q=%lld n_kv=%lld swa=%lld extent=%lld threads=%d pool=%d nodes=%d compute_MiB=%.1f us_mean=%.1f us_min=%.1f gflops=%.1f\n",
           c.backend.c_str(), c.variant.c_str(), (long long) n_q, (long long) n_kv, (long long) c.swa, (long long) E,
           ggml_backend_is_cpu(backend) ? c.n_threads : 0, c.pool ? 1 : 0, ggml_graph_n_nodes(gf), compute_bytes/(1024.0*1024.0), mean, mn,
           flops > 0 ? flops/(mn*1e-6)/1e9 : 0.0);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);
    ggml_backend_free(backend);
    if (tp) ggml_threadpool_free(tp);
    return 0;
}

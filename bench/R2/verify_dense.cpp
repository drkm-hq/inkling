// verify_dense: numerical check that the Metal dense (-fa off) attention chain of src/models/inkling.cpp
// matches the CPU GGML_OP_FLASH_ATTN_EXT_BANDED reference on identical inputs (the question raised by
// the ROCm "-fa 0 degrades quality" report in PR #25731). Prints the NMSE and max abs diff.
//
// usage: verify_dense <n_q> <n_kv> [--swa N] [--extent E]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> rnd(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}

static void set_f16(ggml_tensor * t, const std::vector<float> & f) {
    std::vector<ggml_fp16_t> h(f.size());
    ggml_fp32_to_fp16_row(f.data(), h.data(), f.size());
    ggml_backend_tensor_set(t, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <n_q> <n_kv> [--swa N] [--extent E]\n", argv[0]); return 1; }
    const int64_t n_q = atoll(argv[1]), n_kv = atoll(argv[2]);
    int64_t swa = 0, E = 1024;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--swa") swa = atoll(argv[i+1]);
        else if (k == "--extent") E = atoll(argv[i+1]);
    }
    const int64_t D = 128, H = 32, Hkv = 8;
    const int64_t kv_size = ((n_kv + 255)/256)*256;

    // logical data (shared by both graphs)
    auto qd   = rnd((size_t) D*H*n_q, -1.0f, 1.0f, 1);
    auto kd   = rnd((size_t) D*Hkv*kv_size, -1.0f, 1.0f, 2);   // row j = cell j: [D*Hkv] (head-major within the row)
    auto vd   = rnd((size_t) D*Hkv*kv_size, -1.0f, 1.0f, 3);
    auto reld = rnd((size_t) E*H*n_q, -0.25f, 0.25f, 4);
    // round K/V/rel through f16 where the caches are f16, so both graphs see identical values
    {
        std::vector<ggml_fp16_t> h(kd.size());
        ggml_fp32_to_fp16_row(kd.data(), h.data(), kd.size()); ggml_fp16_to_fp32_row(h.data(), kd.data(), kd.size());
        ggml_fp32_to_fp16_row(vd.data(), h.data(), vd.size()); ggml_fp16_to_fp32_row(h.data(), vd.data(), vd.size());
    }
    std::vector<float> maskd((size_t) n_kv*n_q);
    std::vector<int32_t> idxd((size_t) n_kv*n_q);
    for (int64_t i = 0; i < n_q; ++i) {
        const int64_t p = n_kv - n_q + i;
        for (int64_t j = 0; j < n_kv; ++j) {
            const bool vis = j <= p && (swa == 0 || p - j < swa);
            maskd[i*n_kv + j] = vis ? 0.0f : -INFINITY;
            const int64_t d = p - j;
            idxd[i*n_kv + j] = (int32_t) (i*(E + 1)) + ((d >= 0 && d < E) ? (int32_t) d : (int32_t) E);
        }
    }
    // transposed V storage: vt[(h*D + d)*kv_size + j] = V[j][h*D + d]
    std::vector<float> vtd((size_t) D*Hkv*kv_size);
    for (int64_t j = 0; j < kv_size; ++j)
        for (int64_t c = 0; c < D*Hkv; ++c)
            vtd[c*kv_size + j] = vd[j*D*Hkv + c];

    ggml_backend_load_all();

    auto run = [&](const char * bname, bool banded, std::vector<float> & out) -> bool {
        ggml_backend_t backend = ggml_backend_init_by_name(bname, nullptr);
        if (!backend) { fprintf(stderr, "no backend %s\n", bname); return false; }
        if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, 12);

        ggml_init_params ip = { 32*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx_in = ggml_init(ip);
        ggml_tensor * q    = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F32, D, H, n_q, 1);
        ggml_tensor * kc   = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, D*Hkv, kv_size);
        ggml_tensor * vc   = banded ? ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, D*Hkv, kv_size)
                                    : ggml_new_tensor_2d(ctx_in, GGML_TYPE_F16, kv_size, D*Hkv);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx_in, banded ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_q, 1, 1);
        ggml_tensor * rel  = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, E, H, n_q);
        ggml_tensor * idx  = banded ? nullptr : ggml_new_tensor_1d(ctx_in, GGML_TYPE_I32, n_kv*n_q);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return false; }

        ggml_backend_tensor_set(q, qd.data(), 0, qd.size()*sizeof(float));
        set_f16(kc, kd);
        set_f16(vc, banded ? vd : vtd);
        if (banded) set_f16(mask, maskd); else ggml_backend_tensor_set(mask, maskd.data(), 0, maskd.size()*sizeof(float));
        ggml_backend_tensor_set(rel, reld.data(), 0, reld.size()*sizeof(float));
        if (idx) ggml_backend_tensor_set(idx, idxd.data(), 0, idxd.size()*sizeof(int32_t));

        ggml_init_params gp = { ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * k = ggml_view_4d(ctx, kc, D, Hkv, n_kv, 1,
            ggml_row_size(GGML_TYPE_F16, D), ggml_row_size(GGML_TYPE_F16, D*Hkv), ggml_row_size(GGML_TYPE_F16, D*Hkv*kv_size), 0);
        ggml_tensor * v = banded
            ? ggml_view_4d(ctx, vc, D, Hkv, n_kv, 1, ggml_row_size(GGML_TYPE_F16, D), ggml_row_size(GGML_TYPE_F16, D*Hkv), ggml_row_size(GGML_TYPE_F16, D*Hkv*kv_size), 0)
            : ggml_view_4d(ctx, vc, n_kv, Hkv, D, 1, ggml_row_size(GGML_TYPE_F16, kv_size*D), ggml_row_size(GGML_TYPE_F16, kv_size), ggml_row_size(GGML_TYPE_F16, kv_size*D*Hkv), 0);
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        const float scale = 1.0f/float(D);
        ggml_tensor * res;
        if (banded) {
            ggml_tensor * rel_fa = ggml_reshape_4d(ctx, rel, E, H, n_q, 1);
            res = ggml_flash_attn_ext_banded(ctx, qp, k, v, mask, rel_fa, scale, E);
            ggml_prec_set_acc(res, GGML_PREC_F32);
            res = ggml_reshape_2d(ctx, res, D*H, n_q);
        } else {
            ggml_tensor * qs = ggml_scale(ctx, qp, scale);
            ggml_tensor * kq = ggml_mul_mat(ctx, k, qs);
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            ggml_tensor * r = ggml_pad(ctx, rel, 1, 0, 0, 0);
            r = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));
            r = ggml_reshape_2d(ctx, r, H, (E + 1)*n_q);
            ggml_tensor * kq_b = ggml_get_rows(ctx, r, idx);
            kq_b = ggml_reshape_3d(ctx, kq_b, H, n_kv, n_q);
            kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));
            kq = ggml_add(ctx, kq, kq_b);
            kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f, 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
            res = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), D*H, n_q);
        }
        ggml_build_forward_expand(gf, res);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) {
                fprintf(stderr, "%s: unsupported op %s\n", bname, ggml_op_desc(ggml_graph_node(gf, i)));
                return false;
            }
        }
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "graph alloc failed\n"); return false; }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return false; }
        ggml_backend_synchronize(backend);
        out.resize((size_t) D*H*n_q);
        ggml_backend_tensor_get(res, out.data(), 0, out.size()*sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_in);
        ggml_backend_free(backend);
        return true;
    };

    std::vector<float> a, b, c;
    if (!run("CPU",  true,  a)) return 1;  // reference: CPU banded op (the PR's own semantics)
    if (!run("MTL0", false, b)) return 1;  // Metal dense chain
    if (!run("CPU",  false, c)) return 1;  // CPU dense chain (isolates Metal kernels from the graph construction)
    auto cmp = [&](const char * name, const std::vector<float> & x, const std::vector<float> & y) {
        double num = 0, den = 0, mx = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - y[i];
            num += d*d; den += (double) x[i]*x[i]; mx = std::max(mx, std::fabs(d));
        }
        printf("VERIFY %s n_q=%lld n_kv=%lld swa=%lld extent=%lld nmse=%.3e max_abs_diff=%.3e\n",
               name, (long long) n_q, (long long) n_kv, (long long) swa, (long long) E, num/den, mx);
    };
    cmp("metal_dense_vs_cpu_banded", a, b);
    cmp("cpu_dense_vs_cpu_banded",   a, c);
    cmp("metal_dense_vs_cpu_dense",  c, b);
    return 0;
}

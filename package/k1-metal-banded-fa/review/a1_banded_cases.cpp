    // ---- A1 adversarial cases (review of the Metal banded kernel) ----
    {
        const ggml_type F16 = GGML_TYPE_F16, F32 = GGML_TYPE_F32, BF16 = GGML_TYPE_BF16, Q8 = GGML_TYPE_Q8_0, Q4 = GGML_TYPE_Q4_0;
        // A: n_q sweep at the Inkling geometry, unaligned (4100) and aligned (4096) KV
        for (int64_t nq : { 1, 2, 7, 20, 64, 512, 1024 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, nq, 4100, 1024, 1, F16, F32));
        }
        for (int64_t nq : { 20, 64, 512, 1024 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, nq, 4096, 1024, 1, F16, F32));
        }
        // B1: n_kv sweep around the kernel block sizes, prefill ubatch
        for (int64_t kv : { 512, 513, 575, 576, 577, 639, 640, 641, 1023, 1024, 1025, 1535, 1536, 1537, 2047, 2048, 2049,
                            4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, kv, 1024, 1, F16, F32));
        }
        for (int64_t kv : { 65535, 65536, 65537, 70000 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 8, 2, 512, kv, 1024, 1, F16, F32));
        }
        // B2: n_kv sweep, decode
        for (int64_t kv : { 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
                            65535, 65536, 65537, 70000 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 1, kv, 1024, 1, F16, F32));
        }
        // C: E = 512 (SWA layers): window mask and the band edge at 512 under a causal mask
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512,  512, 512, 2, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 1024, 512, 2, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 1536, 512, 2, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 8192, 512, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 1024, 512, 2, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 8192, 512, 1, F16, F32));
        // D: n_kv < E
        for (int64_t kv : { 1, 7, 100, 500, 1000 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 1, kv, 1024, 1, F16, F32));
        }
        for (int64_t kv : { 64, 100, 500, 1000 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 64, kv, 1024, 1, F16, F32));
        }
        // F: q_offset other than n_kv - n_q (the mask follows the offset)
        for (int64_t qo : { 0, 4096, 7780, 13192, 1 << 20 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 8192, 1024, 1, F16, F32, false, 0, qo));
        }
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 0, F16, F32, false, 0, -300));
        for (int64_t qo : { 0, 1023, 1024, 1 << 20 }) {
            test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 1, 4096, 1024, 0, F16, F32, false, 0, qo));
        }
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, F16, F32, false,   0, 2048));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 8192, 1024, 1, F16, F32, false, 239,    0));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  20, 4100, 1024, 1, F16, F32, false,   0, 1000));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 1, F16, F32, false,  60, 1000));
        // G: no mask
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 8192, 1024, 0, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 8209, 1024, 0, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 0, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  64, 4100, 1024, 0, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 0, F16, F32));
        // H: sliding window 512 under E = 1024
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 2, F16, F32, false, 0, INT64_MIN, 512));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 8192, 1024, 2, F16, F32, false, 0, INT64_MIN, 512));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 2, F16, F32, false, 0, INT64_MIN, 512));
        // I: head configurations
        test_cases.emplace_back(new test_flash_attn_ext_banded(128,  1, 1, 512, 4096, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128,  1, 1,   1, 4096, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128,  8, 8, 512, 2048, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 16, 1, 512, 2048, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded( 64, 32, 8, 512, 4096, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded( 64, 32, 8,   1, 4096, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded( 64, 32, 8,   7, 4100, 1024, 1, F16, F32));
        // J: KV types
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 1, BF16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, BF16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 1, BF16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 8209, 1024, 1, BF16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4100, 1024, 1, Q8,   F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 1, Q8,   F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 8209, 1024, 1, Q8,   F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 1, Q8,   F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 1, Q4,   F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, Q4,   F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, F32,  F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 1024, 1024, 1, F32,  F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  64, 1024, 1024, 1, F32,  F32));
        // K: bias table types
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 1, F16, F16));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 1, F16, BF16));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, F16, F16));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, F16, BF16));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  64, 4100, 1024, 1, F16, BF16));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 1, F16, F16));
        // L: batch dimension
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  64, 1024, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 1));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  64, 1024, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 2));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 2048, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 2));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 2048, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 1));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 2048, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 2));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 2048, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 1));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 2049, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 2, 2));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,  64, 1024, 1024, 1, F16, F32, false, 0, INT64_MIN, 0, 3, 1));
        // M: strided table and KV at the Inkling geometry
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8, 512, 4096, 1024, 1, F16, F32, true));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, F16, F32, true));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 4100, 1024, 1, F16, F32, true));
        // N: the n_kv_max hint (Metal sparse vec path), never set by the model but read by the backend
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 4096, 1024, 1, F16, F32, false, 0, INT64_MIN,   0, 1, 1, 4096));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   7, 2048, 1024, 1, F16, F32, false, 0, INT64_MIN,   0, 1, 1, 2048));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   1, 8192,  512, 2, F16, F32, false, 0, INT64_MIN, 512, 1, 1,  512));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128, 32, 8,   2, 1024, 1024, 1, F16, F32, false, 0, INT64_MIN,   0, 1, 1, 1024));
        // O: large KV (offset arithmetic)
        test_cases.emplace_back(new test_flash_attn_ext_banded(128,  8, 2,   1, 1048576, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128,  8, 2,  32, 1048576, 1024, 1, F16, F32));
        test_cases.emplace_back(new test_flash_attn_ext_banded(128,  8, 2, 512,  262144, 1024, 1, F16, F32));
    }
    // ---- end A1 cases ----


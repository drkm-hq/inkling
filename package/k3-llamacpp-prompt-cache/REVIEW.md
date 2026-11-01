# A3 adversarial review: `server-hybrid-prompt-cache` (K3)

Reviewed: branch `server-hybrid-prompt-cache` at `5edd1f152` (one commit on `pr-25731` = `3582c5ee0`), the package
`package/k3-llamacpp-prompt-cache/` (README.md, PR.md, `patches/0001-*.patch`, `bench/`). Reviewer branch
`a3-review` in `upstream/llama.cpp-a3` (commit `795968add`: the adversarial tests, review only), the upstream-fit
branch `a3-upstream-master` in `upstream/llama.cpp-a3-master` (`origin/master` `2e0d86beb` + the patch =
`f1466e736`). Raw logs under `review/`. Nothing pushed, K3's branch untouched.

## Verdict per claim

| claim | verdict | evidence |
|---|---|---|
| Mechanism: hybrid `seq_pos_min` is the recurrent tail; the `n_swa` back-off makes the gate fire on every request and the checkpoint search fail; `n_past = 0` | CONFIRMED | code (section 1), K3's test on the baseline `(0, 93)` instead of `(66, 27)`, my 11 baseline cases all `(0, n)` or a fallback to the `n_ubatch + 4` checkpoint |
| The predicate `pos_min_is_tail` (FULL/RS) is correct for every memory type the server can see | CONFIRMED | section 1: the recurrent tail dominates `max(attn_min, recr_min)`, every rewind still fails the gate, a strict extension passes exactly when the tail is `pos_next - 1`; PART models keep the back-off |
| Nothing changes for non-hybrid models (same decisions, same logs) | CONFIRMED | dense (stories260K) and SWA-PART (tinygemma3, `n_swa 4096`, prompts past the window): identical `(cache_n, prompt_n)`, identical top-5 logprobs (0.000 nats), identical decision lines on fixed, baseline and master+patch |
| Strict extensions reuse the live state, rewinds restore the newest checkpoint before the edit, outputs match the uncached reference | CONFIRMED | 22/22 adversarial cases pass on the fixed build and on master+patch; on the small model, with identical ubatch splits, the checkpoint path is bit-exact (0.000 nats) against the uncached evaluation; on Inkling the live state equals its own checkpoint to 0.0004 nats (section 4) |
| Patch applies to upstream `master` and the test passes there | CONFIRMED | `git am` clean on `2e0d86beb`, `llama-server` built, `test_completion_cache_hybrid_memory[0/64]` 2 passed, my 22 cases identical to the fixed build |
| The regression test is CI-ready | PLAUSIBLE | passes 3/3 here; the `5e-2` logprob tolerance on 4-token suffixes of a Q4_K_M model is thin for other backends (2-token suffixes already deviate by 0.10-0.19 nats on Metal, unrelated to the fix) |
| Measurements (`-ub 512`: 3.6 s -> 1.43 s, 4.27 -> 1.42, 30.3 -> 1.90; chat turns 3.1-3.5 s -> 0.8 s) | PLAUSIBLE | like-for-like flags and ubatch, but the after-run ran under 2.8x the load (load1 6.5 vs 17.9) and its uncached reference is 15% slower (29.9 vs 34.5 s); the token counts are deterministic and reproduced; see section 5 and the real-model section |
| Greedy divergences in the extension cases are near-ties / accumulation-order drift, not a wrong state | PLAUSIBLE (section 4) | all margins at a divergence <= 0.45 nats; the live state equals its own checkpoint to 0.0004 nats; checkpoint-only paths (no live state, the baseline's mechanism) diverge at least as early (tokens 9-32, ~1 nat step-0 deviations) and even identical-split restores diverge at token 14 on this model; two of ten cached 64-token continuations loop after diverging (none of seven uncached); the `-ub 256` control and an 8-prompt loop-rate sample (section 4b) were queued but not run before the review closed |
| PR.md states accurately what the human must rewrite | PLAUSIBLE with one gap | it covers the commit message and description, but not that the AI trailer must be `Assisted-by:` (AGENTS.md forbids `Co-authored-by:` for the assistant) |
| "Chunked reuse (`--cache-reuse`) stays unavailable for FULL memories" (README follow-up 3) | BROKEN (the statement, not the code) | the server only disables it when the memory cannot shift; a hybrid without SWA accepts it and shifts the attention cache while the recurrent state keeps the removed chunk (pre-existing, all three binaries) |

Overall: the change is correct and minimal; no defect found in the code. Required before submission: the items in
section 6 (all documentation/process, one test-robustness item).

## 1. Code review

The change (`tools/server/server-context.cpp:4104-4112`):

```
const bool pos_min_is_tail = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                             ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
const auto pos_min_thold = std::max(0, n_swa > 0 && !pos_min_is_tail ? pos_eval - n_swa + 2 : pos_eval);
```

`pos_min_thold` is used twice: the live-state gate `pos_min >= pos_min_thold` (`:4160`) and the checkpoint predicate
`cur.pos_min < pos_min_thold || cur.pos_min == 0` (`:4174`). Facts the correctness rests on (all verified in the
tree at `3582c5ee0`, unchanged on `origin/master`):

- `llama_memory_hybrid[_iswa]::seq_pos_min = max(mem_attn, mem_recr)` (`src/llama-memory-hybrid-iswa.cpp:178`,
  `src/llama-memory-hybrid.cpp:173`); `llama_memory_recurrent::seq_pos_min` is the min over the cells of the seq
  (`src/llama-memory-recurrent.cpp:391`), one live cell per seq whose pos is the last decoded position. The SWA
  cache's min is <= its max = that same position, so for a hybrid `pos_min` is always the recurrent tail and carries
  no information about the SWA window start.
- A token enters `slot.prompt.tokens` only when it is added to a decode batch (`handle_last_sampled_token`,
  `:589-641`; the prompt loop `:4412`), so the tail is `n_tokens - 1` whenever the slot is idle: the final sampled
  token is not kept (K3's "[:-1]" is right).
- `common_context_can_seq_rm` (`common/common.cpp:1596`) returns RS iff `n_rs_seq > 0`, else FULL iff the partial
  `seq_rm` fails. The only `seq_rm` that fails on a partial range is the recurrent one
  (`src/llama-memory-recurrent.cpp:186-217`: `0 < p0 <= cell.pos < p1` without rollback capacity); `llama_kv_cache`
  and `llama_kv_cache_iswa` never refuse. So FULL/RS <=> the memory contains a recurrent part, today. (The predicate
  conflates "cannot partially remove" with "pos_min is a tail"; a future non-recurrent FULL memory would need its
  own case. Worth one sentence in the PR.)
- Checkpoints of a FULL/RS memory record `(pos_min, pos_max) = (tail, tail)` and `n_tokens = tail + 1`
  (`create_checkpoint`, `:2680-2745`; the logs show `pos_min = pos_max = n_tokens - 1`). `PARTIAL_ONLY` writes the
  SWA cache and the recurrent state, not the base cache (`src/llama-kv-cache.cpp:2130, 2211`); the base cache is
  truncated by the `seq_rm(pos_next, -1)` at `:4269`, which the recurrent part accepts because `p0 > cell.pos` after a
  restore (`pos_next = min(pos_next, max(pos_min + 1, pos_max)) = tail_c + 1`).

Per memory type, with `pos_eval = pos_next` (new tokens) or `pos_next - 1` (identical prompt):

| memory | type | `n_swa` | threshold before -> after | decision |
|---|---|---|---|---|
| plain KV (dense) | PART | 0 | `pos_eval` -> `pos_eval` | unchanged |
| SWA attention only (gemma-style) | PART | > 0 | `pos_eval - n_swa + 2` -> same | unchanged: `!pos_min_is_tail` keeps the back-off; the window-start protection (a rewind past the pruned window must restore a checkpoint) stays |
| pure recurrent (mamba/rwkv) | FULL | 0 | `pos_eval` -> `pos_eval` | unchanged |
| hybrid without SWA (jamba, nemotron-h, qwen3-next, LFM2 as-is) | FULL/RS | 0 | `pos_eval` -> `pos_eval` | unchanged |
| hybrid iswa (Inkling, LFM2 with a window, LFM3) | FULL/RS | > 0 | `pos_eval - n_swa + 2` -> `pos_eval` | strict extension: `tail = pos_next - 1 < pos_next` passes, state used as-is. Identical prompt: `tail = pos_eval` fails -> checkpoint `tail_c < pos_eval`. Any rewind (`pos_eval <= tail`) fails -> checkpoint before the first changed token, else full re-processing. This is exactly the `n_swa == 0` hybrid behaviour. |
| hybrid iswa with `--swa-full` | FULL/RS | 0 | unchanged | |

What the old gate protected and whether the new one still does:

- SWA-PART window start advanced past the rewind point: still protected, the back-off is untouched for PART.
- Hybrid iswa, the SWA cache after a rewind: never reachable through the live state (every rewind fails the tail
  test), only through a checkpoint whose SWA cache holds the full window for `tail_c + 1` (the cache keeps the
  cells with `pos >= pos_max + 1 - n_swa + 1`, `src/llama-kv-cache.cpp:1052`, which is what position `tail_c + 1`
  attends to). The old gate's extra conservatism (a checkpoint at least `n_swa - 2` tokens back) bought nothing.
- `seq_rm` abort hazard: `common_context_seq_rm` aborts if the recurrent part refuses. After the gate the server calls
  `seq_rm(pos_next, -1)`; with the new threshold the live state is kept only when `tail < pos_next`, a checkpoint only
  when `tail_c < pos_eval <= pos_next`, so `p0 > cell.pos` in both cases and the recurrent part is a no-op. Same for
  the speculative checkpoint path (`:4898`).
- `[TAG_PROMPT_LOGITS]` (`n_past--` when nothing would be evaluated) is not reached for FULL/RS: the identical prompt
  always fails the gate.

Other paths: `cache_prompt: false` sets `n_past = 0` before the gate (unchanged). `--slot-save` / restore: the file
holds the live state + the appended checkpoints; the restored slot behaves like a live one (verified, case 6).
Context shift: allowed for hybrids (`get_can_shift` is the attention part's); the recurrent `seq_rm` of a middle range
is a no-op and `seq_add` moves the tail, so a shifted slot stays consistent with the gate (verified, no abort, case 7).
`n_keep` only feeds the shift. `--cache-reuse`: disabled for hybrid-iswa (`llama_kv_cache_iswa::get_can_shift`
requires equal base/SWA sizes, so "cache_reuse is not supported by this context"), accepted for a hybrid without
SWA where it is an approximation (case 8). Multi-slot and unified KV: `seq_pos_min` is per sequence in both layouts
(case 2). Draft model: `ctx_dft` has its own type and checkpoints, untouched. Checkpoint creation after the gate
(`:4303-4316`, `:4425-4495`) is unchanged; a strict extension now starts from the live state and still gets the
checkpoints at `n - (n_ubatch + 4)` and `n - 4` when those fall in the evaluated range, plus one at the start of
any batch that ends within `n_ubatch` of the prompt end (`near_prompt_end`).

Nits in the text (not the code): PR.md says the fallback checkpoint exists "only when `n_ubatch + 4 > n_swa - 2`";
the exact condition is `tail_c = n - n_ubatch - 5 < n - n_swa + 2`, i.e. `n_swa < n_ubatch + 7` (one off at the
boundary, harmless for 512/512 and 512/256). The commit comment's claim "the checkpoint search cannot succeed either"
holds only when `n_swa >= n_ubatch + 7`; at `-ub 512` it does succeed 516 tokens back, as K3's own before-numbers show.

## 2. Tests

Environment: Python 3.12 venv (`temporary directory/venv-k3`), `LLAMA_CACHE=temporary directory/llama-cache` (offline, the LFM2 CI
model and the other presets already cached), `tools/server/tests/tmp` present. Binaries: fixed = `build-k3`
(built from `7251878de`, the pre-amend commit: identical code, the amend only shortened the comment; see section 3),
baseline = `build-k3-baseline` (`3582c5ee0`), master = `build-a3m` (`f1466e736`). Ports: the harness takes
`LLAMA_SERVER_TEST_PORT_BASE` (conftest overrides `PORT`); my first subset run shared 8080 with another run and was
repeated on 8700.

K3's test (`review/a3-k3test-*.log`):

| binary | `[0]` | `[64]` |
|---|---|---|
| fixed | PASSED | PASSED (6/6 runs incl. 5 repeats, `review/a3-k3test-fixed-repeat*.log`) |
| baseline | PASSED | FAILED at the strict extension: `(cache_n, prompt_n) = (0, 93)`, expected `(66, 27)` |
| master + patch | PASSED | PASSED |

K3's subset (`test_completion test_slot_save test_ctx_shift test_speculative test_basic test_chat_completion
test_kv_keep_only_active`, `review/a3-subset-*.log`): fixed 140 passed, 1 failed, 1 skipped; baseline 139 passed,
2 failed, 1 skipped. The failure common to both is `test_completion_stream_with_openai_library_stops`, whose server
needs `bartowski/Phi-3.5-mini-instruct-GGUF` and the cache is offline (environment, not the change); the baseline's
second failure is K3's new `[64]`. The skip is upstream's `test_cache_vs_nocache_prompt`.

K3's own `bench/test-subset-{fixed,baseline}.log` (4 xdist workers, 141 items, 22:48-22:49) end with "2 failed,
138 passed, 1 skipped" on BOTH binaries, the two failures being the new test's `[0]` and `[64]` in an earlier
version with a `1e-3` logprob tolerance (`-1.1977 == -1.2001 +- 1e-3`); the committed `5e-2` version was then run
on its own (`bench/test-hybrid-*.log`, 22:50). The README's "138 passed / 1 skipped plus the new test" is true of
the other tests but should say this, and the tolerance history is itself the argument for the robustness item in
section 6.

Adversarial cases (`tools/server/tests/unit/test_a3_hybrid_cache.py` on `a3-review`, 22 cases, LFM2-test-ci-80M
`-ub 32 -b 64 --cache-ram 0 --temp 0`, every cached request's top-5 logprobs of the first predicted token checked
against the same prompt without the cache; `review/a3-cases-*.log`, per-case `review/a3-cases/<case>-<binary>.json`
and `.server.log`). Tuples are `(cache_n, prompt_n)` per request, in order; fixed and master+patch are identical in
every case.

| case | fixed = master | baseline |
|---|---|---|
| 1 rewind at token 210 of a 2000-token prompt, `n_swa 0`: prime 200, extend, edit (checkpoint at 196 exists), fresh prime, edit (no early checkpoint) | (0,200) (200,1800) (196,1804) (0,2000) (0,2000) | same |
| 1 same, `n_swa 64` | same | (0,200) (0,2000) (0,2000) (0,2000) (0,2000) |
| 2 two slots alternating, `n_swa 64`, non-unified and `--kv-unified` (slot, cache_n, prompt_n) | (0,0,300) (1,289,11) (0,300,25) (1,300,25) (0,325,20) (1,321,4) | (0,0,300) (1,0,300) (0,0,325) (1,0,325) (0,0,345) (1,0,325) |
| 3 `cache_prompt: false` between cached requests, `n_swa 0` / `64` | (0,300) (0,330) (330,20) (0,350) (0,350) | `0`: same; `64`: (0,300) (0,330) (0,350) (0,350) (0,350) |
| 4 shrink: prime 300, prefix 298, prefix 250, 298 again, prefix 296, `n_swa 0` / `64` | (0,300) (296,2) (0,250) (250,48) (294,2) | `0`: same; `64`: (0,300) (0,298) (0,250) (0,298) (0,296) |
| 5 window smaller than the extension, `n_swa 4 / 8 / 16`: prime 100, +40, identical, +1, +1, rewind into the extension | (0,100) (100,40) (136,4) (140,1) (141,1) (104,28) | `8`, `16`: (0,100) (64,76) (104,36) (104,37) (105,37) (106,26) (the `n_ubatch + 4` fallback); `4`: the same fallback, its run stopped at a 4th/5th top-5 order swap (-2.648 vs -2.658 nats) before recording |
| 6 slot save on slot 0, restore into slot 1, extend, identical (`n_swa 64`) | (0,0,300) (1,300,30) (1,296,4) | (0,0,300) (1,0,330) (1,0,300) |
| 7 context shift on the hybrid (`-c 256`, 200 + 120 generated, `n_keep 10`), then two cached requests | shift logged (`n_discard = 122`), no abort, (0,200) (0,100) | same |
| 8 `--cache-reuse 16`, a 50-token chunk removed: `n_swa 0` | (0,300) (250,20): chunk shifted, recurrent state keeps the chunk, top-1 equal, logprobs off by 0.15 | same (pre-existing) |
| 8 same, `n_swa 64` | "cache_reuse is not supported by this context", (0,300) (0,270) | same |
| 9 dense stories260K (plain KV): prime, +30, identical, rewind, shrink | (0,300) (300,30) (329,1) (320,10) (249,1); no checkpoint/forcing lines in the log | identical, logprobs identical |
| 10 SWA-PART tinygemma3 (`n_swa 4096`, prompts of 4300 past the window): prime, +30, identical, rewind, shrink to 3000 | (0,4300) (4300,30) (4329,1) (4320,10) (0,3000); one "forcing full" line (the shrink) | identical, logprobs identical |
| 11 small suffixes (12 tokens) from the live state and from a checkpoint | (0,128) (128,12) (128,4) (128,12); deviation 0.004 / 0.007-0.009 nats | `64`: (0,128) (0,140) (0,132) (0,140) |
| 12 identical ubatch splits: uncached X (164 tokens, checkpoints at 128 and 160), X' = X[:128] + 36 other tokens cached, X' again | (0,164) (0,164) (128,36) (160,4); X' vs uncached X': **0.000** nats on all top-5; X' again vs X': 0.000 (`n_swa 0`), 3.8e-6 (`n_swa 64`) | `64`: (0,164) (0,164) (0,164) (0,164) |

The real-model driver (`review/a3_gpu_run.py`) was dry-run on the same small model at `-ub 512 -c 16384`
(`review/dryrun/dryrun-lfm2.jsonl`, p_8k = 11641 LFM2 tokens): the 8K-length versions of case 12 agree at 0.000
nats over the full generation (X' through the checkpoint at 7876 vs uncached, 47 tokens; the strict extension from
the live state vs the same request through its own checkpoint, 13 tokens; two uncached runs, 64 tokens), and the
4-turn chat with the edited turn 3 gives prompt_n 3169 / 77 / 81 / 52 with cache_n 0 / 3165 / 3165 / 3242 (turn 3
restores the checkpoint at the start of user message 2).

Output agreement: every cached request on the fixed build has the same top-1 token as its uncached reference and
the same top-5 id set. Deviations on suffixes of >= 4 tokens are <= 0.022 nats; suffixes of 1-2 tokens deviate by
0.04-0.19 nats (cases 4, 5), and the same numbers appear on the baseline for `n_swa 0` (e.g. shrink-by-2 again:
0.191 on both), so they are the numerics of a 1-2 row matmul versus a 32-row ubatch on this Q4_K_M model, not the
change. Case 12 removes the split difference and gets exact equality, which is the strongest statement available
that a restored checkpoint plus the evaluated suffix equals the uncached evaluation. No crash, abort or
"pos_min == -1" in any log.

## 3. Upstream fit

- `patches/0001-*.patch` is byte-identical to `git format-patch -1 5edd1f152`. `git am` onto `origin/master`
  (`2e0d86beb`, fetched 2026-10-29) applies cleanly (`f1466e736`); the gate code is identical on master although
  `server-context.cpp` differs elsewhere by 136/23 lines since `pr-25731`. `llama-server` built there
  (`build-a3m`, Release, Metal; `review/../llama.cpp-a3-master/build-a3m-build.log`), K3's test 2 passed, my 22
  cases identical to the fixed build (`review/a3-cases-master.log`).
- The binary K3 measured and tested (`build-k3`, `build 11984 (7251878de)`) is the pre-amend commit; the amend
  (`5edd1f152`, committer 22:56:53) changed only the comment above `pos_min_is_tail` (7 lines -> 3). The code is
  the same, but the bench meta rows record `commit 5edd1f152` for all three runs (the driver reads the worktree
  HEAD, not the binary), and the baseline run's row says the same although it ran `3582c5ee0`. Say so in RESULTS.md
  or rebuild before any re-measurement.
- CONTRIBUTING.md / AGENTS.md: AI-generated code is allowed with disclosure; AI-written PR descriptions, commit
  messages and reviewer responses are prohibited; a commit made on the user's behalf must carry
  `Assisted-by: <assistant>` and must NOT use `Co-authored-by:` for the assistant (AGENTS.md line 96). The commit
  `5edd1f152` carries `Co-Authored-By: Claude Fable 5.1` and `Co-authored-by: fezzik <fezzik@fezzik.dev>` (the fork
  author's identity matches the fork commit (https://github.com/F3zz1k/llama.cpp/commit/371c74a0553663d9d6e28b4ebecac9542f076c19): `fezzik <fezzik@fezzik.dev>`, so that credit is
  right). PR.md's box tells the human to rewrite the commit message and description and to disclose; it does not
  say the assistant trailer must become `Assisted-by:` (or go). Add that line.
- PR.md's related-PR list spot-checked read-only with `gh`: #24035 is OPEN since 2026-06-02, 0 reviews, 6 comments,
  last update 2026-07-27, title as stated; #22932 CLOSED; #24797 ("server: fix checkpoint handling for hybrid memory
  models", CLOSED 2026-06-19) and #22826 (CLOSED) are not mentioned and are worth a look before posting, since a
  reviewer may ask how this differs from them.
- The new preset is picked up by `ServerPreset.load_all()` (it iterates the class' static methods), so CI will
  download `ggml-org/LFM2-test-ci-80M` (50 MB) by itself.
- The port is faithful to the fork commit: F3zz1k's `pos_min_thold = max(0, pos_next - (pos_min_is_tail ? 0 :
  n_swa) - (has_new_tokens ? 0 : 1))` equals `pos_eval` for tails; K3 keeps upstream's current PART form
  (`pos_eval - n_swa + 2`) and uses `pos_eval` for tails. The fork's second hunk (a "restore-continue" fast path)
  is rightly left out.

## 4. Real model (Inkling-Small UD-IQ3_XXS, fixed build)

One hold of 8 minutes (`review/a3_gpu_turn.sh`, lock taken 04:58:08Z, released 05:06:15Z; `review/a3_gpu_run.py`,
rows in `review/a3-fixed-ub512.jsonl`, server log `review/a3-fixed-ub512-server.log`, tables via
`review/a3_gpu_summarize.py` and `review/a3_gpu_compare.py`). Flags as K3: `-np 1 --no-kv-unified -ngl 99 -fa 0
-c 70000 --jinja --temp 0 -ub 512`, `n_predict 64`, `n_probs 5`, streamed; model load 8.0 s (warm), RSS 91.4 GiB,
load1 11.1 at start / 14.9 at the end; p_8k = 8218 tokens, extension 200, p_2k = 2072 (K3's counts).

K3's cases reproduced, same `(cache_n, prompt_n)` as K3's after-run:

| request | prompt_n / cache_n | prompt_ms | TTFT | K3 after-run |
|---|---|---|---|---|
| p_8k, no cache | 8218 / 0 | 31.47 s | 31.54 s | 34.47 s |
| p_8k + 200, no cache | 8418 / 0 | 29.05 s | 29.06 s | 33.79 s |
| p_8k cached (slot held p_8k + 200 + gen): checkpoint 7902 | 316 / 7902 | 1.79 s | 1.80 s | 1.83 s |
| (a) p_8k + 200 cached: checkpoint 8214 | 204 / 8214 | 1.29 s | 1.30 s | 1.43 s |
| p_8k cached again: checkpoint 8214 | 4 / 8214 | 0.16 s | 0.17 s | 0.23 s |
| (a2) p_8k + 63 generated + 200 cached: live state | 200 / 8281 | 1.21 s | 1.21 s | 1.42 s |
| (c) last 100 tokens replaced cached: checkpoint 7965 | 253 / 7965 | 1.58 s | 1.59 s | 1.90 s |

Server log: no "forcing full prompt re-processing" line for any cached request; the strict extension has no
restore line (`cached n_tokens = 8281, memory_seq_rm [8281, end)`), the others `restored context checkpoint` at
7902 / 8214 / 8214 / 7965 as above.

4-turn chat on p_2k, turn 3 replacing turn 2's user text (`/v1/chat/completions`, streamed, `reasoning_effort
none`): turn 1 2085 / 0, 9.50 s, TTFT 10.2 s; turn 2 18 / 2101, 0.60 s, TTFT 0.94 s (strict extension); turn 3
(edited) 22 / 2102, 0.58 s, TTFT 0.83 s (the checkpoint at the start of user message 2, then the new text); turn 4
16 / 2154, 0.62 s, TTFT 0.84 s. Answers: turn 1 "The narrator is Ishmael, and the ship is the Pequod."; the rest
coherent and on topic. RSS 90.30 GiB before turn 1, 91.55 GiB after turn 4; over the whole run the RSS moved
between 90.3 and 92.0 GiB with no trend (the checkpoints, 75 MiB each at 8K, come and go). No growth.

Exactness and divergence (`review/a3_gpu_compare.py`; "dev" = max |dlogprob| over the common top-5 ids, "margin"
= in each path's own top-5, its top-1 minus the other path's token; "KL5" = KL over the renormalised common top-5,
mean over the steps before the divergence):

| pair | first diff | step-0 dev | pre-div mean / max dev | margins (cached / ref) |
|---|---|---|---|---|
| X uncached twice (run-to-run, identical splits) | none (64) | 0.000 | 0.000 / 0.000 | |
| same request, live state vs its own checkpoint 4 before the end (B4) | none (64) | 0.0004 | 0.013 / 0.111 | |
| X' through the checkpoint at 7876 vs uncached X', identical ubatch splits | 14 | 0.068 | 0.311 / 0.851 | 0.162 / 0.017 |
| (c) rewind through checkpoint 7965 vs uncached | none (64) | 0.093 | 0.168 / 0.590 | |
| (a) +200 through checkpoint 8214 vs uncached (K3's B1) | 5 | 0.423 | 0.605 / 0.888 | 0.058 / 0.452 |
| (a2) strict extension, live state vs uncached (K3's B2) | 9 | 0.345 | 0.579 / 2.068 | 0.109 / 0.356 |
| same prompt p_8k through checkpoint 7902 vs uncached | 9 | 1.092 | 0.373 / 1.092 | 0.112 / 0.093 |
| same prompt p_8k through checkpoint 7965 vs uncached | 32 | 0.958 | 0.326 / 1.810 | 0.029 / 0.122 |
| same prompt p_8k through checkpoint 7876 vs uncached | 9 | 0.955 | 0.369 / 0.955 | 0.018 / 0.093 |

Reading:

- The backend is deterministic run-to-run (0.000 over 64 tokens), and the live state is the same state as the
  checkpoint taken 4 tokens earlier plus those 4 tokens (0.0004 at step 0, no divergence in 64 tokens): the
  fix's new path (live-state reuse) is equivalent to the checkpoint path the baseline already relied on.
- Every margin at a divergence is below 0.5 nats (max 0.452), i.e. a near-tie in at least one of the two
  distributions, none above the 0.7-nat line; but the noise floor on this model is such that the line is not
  the right instrument: the three `same prompt through a checkpoint` rows involve no live state at all, only a
  checkpoint restore (the mechanism the baseline uses for every reuse) plus a different ubatch split of the last
  253-342 tokens, and they diverge at tokens 9-32 with step-0 deviations of ~1 nat on individual top-5 entries.
  Even with identical ubatch splits a restored SWA cache diverges at token 14: a single-sequence restore clears
  the sequence and re-inserts its cells wherever `find_slot` places them (`llama_kv_cache::state_read_meta`,
  `src/llama-kv-cache.cpp:2419, 2497, 2507`), so the window's keys sit in other physical cells and the attention
  reductions run in another order; on the small LFM2 model the same effect is 3.8e-6 nats (case 12 with `n_swa
  64`) and exactly 0 without an SWA cache (case 12 with `n_swa 0`, where the checkpoint holds only the recurrent
  state), on Inkling (35 SWA layers, MoE routing) it grows to a greedy divergence. The live path (a2) sits inside
  that floor (step-0 0.35 vs 0.96-1.09 for the checkpoint-only rows; divergence at 9 vs 9-32), and the live path
  touches no cell at all.
- Pending: the uncached-at-`-ub 256` control and a loop-rate sample (section 4b).

4a. Coherence. The answer to the question in p_8k ("who is the narrator and what is the ship") is right in every
cached and uncached path ("Ishmael", "the Pequod"). Two cached continuations degenerate after their divergence
in the 64-token window: p_8k through checkpoint 7902 ("...the Pequod.\n\nBased on the Chapter 1 text, Based on
the Chapter 1 text, Based\n\nBasedBasedBased...") and the live-state extension (a2) ("...a hermit and a crucifix
were together. There stand his trees, each with a hollow trunk, as if a hermit and a crucifix were together.
There stand his trees..."), while the uncached references in this sample ramble (the p_8k reference continues
"The user wants me to identify the narrator...") but do not loop. The other eight cached continuations are
coherent, including the second live-state extension (B4). Two loops in ten cached runs against none in seven
uncached runs, with greedy decoding of a 3-bit quant and no repetition penalty, is not a finding yet; section 4b
samples it.

4b. Noise floor and loop rate: NOT RUN. The optional second hold was still last in the queue (A2, M1, A3 behind
K2's 40-minute chain) when the review closed at 07:15Z, two hours after it was queued. `review/a3_gpu_noise.py`
evaluates p_8k, p_8k + 200 and X without the cache at `-ub 256` (against the `-ub 512` uncached rows: accumulation
order only, no cache), X + 4 generated tokens (other batch breaks at the same ubatch), then at `-ub 512` eight
p_2k + 100-token extensions each once uncached and once as a strict extension from the live state, with a
repeated-6-gram loop detector on both. The waiter `review/a3_gpu_turn2.sh` (pid 3312, A3 behind K2, K4 and A4 in
`bench/GPU_QUEUE` at 06:11Z) takes the lock by the protocol, caps the hold at 15 minutes and cleans up; the script
was dry-run on the small model (`review/dryrun/dryrun-noise.jsonl`). When it has run: `python3
review/a3_gpu_compare.py review/a3-fixed-ub512.jsonl review/a3-noise-ub256.jsonl` prints the `-ub 256` vs `-ub 512`
rows (the floor with no cache involved) and `grep loop_summary review/a3-noise-ub256.jsonl` the loop counts
(cached vs uncached out of 8). Reading guide: if the uncached `-ub 256` rows diverge from `-ub 512` at similar
tokens with similar step-0 deviations (0.3-1 nat) the live-path numbers in the table above are the ordinary
accumulation-order drift of this model; a loop count for the cached side clearly above the uncached side would
be the one observation that still deserves a closer look (it would point at the SWA-cache restore order or at
the live path, separable by comparing the loop rows' `cache_n`). To cancel the hold instead: `kill 3312` (its
trap removes `A3` from the queue and releases nothing it does not hold).

## 5. Benchmark skepticism

- Like for like: the three runs use the same command line except the binary and `-ub` (verified from the `meta`
  rows and `-server.log` headers: `-np 1 --no-kv-unified -ngl 99 -fa 0 -c 70000 --jinja --temp 0`, `n_batch 2048`),
  `n_predict 64`, greedy, one warm-up request each; the before/after columns are both `-ub 512`.
- Machine load at the start of each run (`load1` in the meta rows): before-ub512 6.5, after-ub512 17.9,
  before-ub256 26.7. The after run was the most loaded of the three where it matters: its uncached p_8k reference
  took 34.5 s against 29.9 s before (+15%), its 64-token generations 11-21 s against 11-17 s. So the absolute
  prompt_ms numbers carry +-15% noise and the after column is handicapped, not favoured; the ratios K3 quotes are
  dominated by the token counts (716 -> 204, 779 -> 200, 8218 -> 253, 550-592 -> 16-23), which are deterministic
  and reproduced here on the small model and in the server logs. Treat "3.6 s -> 1.43 s" as "716 -> 204 tokens";
  the per-token prompt rate (~4 ms/token at 8K, 5-7 ms/token for the short suffixes) is the stable quantity.
- The before-ub256 column has no uncached references (`--quick`), as labelled; its 42-46 s full re-processing
  numbers are 25-35% slower than the -ub 512 full passes of the other runs, which is the smaller ubatch plus load
  26.7, not a property of the baseline.
- Memory: RSS 91.8-94.1 GiB in all runs, 85.8 GiB at the end of the before run; no growth across the chat turns
  in K3's logs. The 64-token decode times in the before/after runs (170-330 ms/token) are 4-5x slower than in the
  before-ub256 run (47-87 ms/token), another sign of contention from other jobs during K3's hold.
- K3's divergence data (after run, `probs` recorded top-3 per step): B1 diverges at token 5 with margins 0.058
  (cached) / 0.452 (reference) nats; B2 at token 9 with 0.109 / 0.356; both below 0.7. But the per-step top-3
  deviations before the divergence are 0.08-0.60 nats (B1) and 0.07-2.07 nats (B2), and even the bit-identical
  rewind C deviates by 0.05-0.53 nats per step, so on this model the logprob noise floor between two different
  ubatch splits is a few tenths of a nat and the margin criterion alone cannot separate drift from a state error;
  the identical-splits checks in section 4 are there for that.

## 6. Defects and required changes before submission

Defects in the code: none found.

Required changes:

1. PR.md: add that the assistant trailer must be `Assisted-by:` (AGENTS.md), not `Co-authored-by:`; keep the
   fezzik credit. Mention #24797 and #22826 in the related list (both closed attempts at the same area).
2. RESULTS.md / README.md: state that the measured fixed binary is `7251878de` (code-identical to `5edd1f152`,
   comment-only amend) and that the baseline run was `3582c5ee0` despite the `commit` field; state the load
   figures (6.5 / 17.9 / 26.7) next to the timings and present the gains as token counts first; say that the
   subset logs predate the committed test (its `1e-3` version fails there on both binaries) or re-run the subset
   with the committed tree.
3. README.md follow-up 3: replace "Chunked reuse stays unavailable for FULL memories" with what the server does
   (disabled only when the memory cannot shift; for a hybrid without SWA the chunk shift is applied and the
   recurrent state keeps the removed tokens), and consider a separate upstream note that `can_cache_reuse`
   should require `COMMON_CONTEXT_SEQ_RM_TYPE_PART`.
4. PR text precision: "the checkpoint search cannot succeed either" -> "cannot succeed unless a checkpoint lies more
   than `n_swa - 2` tokens back, which the default placement provides only when `n_swa < n_ubatch + 7`".
5. Test robustness (recommended): the logprob tolerance `5e-2` on the 4-token suffix steps is thin for other
   backends (the same model deviates by up to 0.19 nats on 2-token suffixes on Metal, baseline included). Keep the
   `(cache_n, prompt_n)` assertions and the top-1 id, loosen the logprob bound (0.2) or compare the cached request
   against a second cached request with the same splits.
6. One sentence in the PR on the assumption behind the predicate: FULL/RS arises only from recurrent memories today.

## 6a. Post-review state (K3's response)

K3 amended the branch to `9b8b6bd93` (`patches/0001-*.patch` regenerated 02:13): `server-context.cpp` is
byte-identical to the reviewed `5edd1f152`; the test's output comparison is now top-1 id equal plus the shared
top-5 candidates (at least 3) within 0.2 nats (item 5); the trailer is `Assisted-by:` with the fezzik
`Co-authored-by:` kept (item 1); PR.md names the trailer rule and #24797, RESULTS.md the `7251878de` /
`3582c5ee0` provenance and the load figures, README.md the real `--cache-reuse` behaviour (items 2-4). The
amended test passes on the fixed binary and on the master build (`review/a3-k3test-v2-*.log`), and the
regenerated patch applies cleanly to `origin/master` (`a3-upstream-master-v2`). Verdicts unchanged.

## 7. Raw logs

- `review/a3-k3test-{fixed,baseline,master}.log`: K3's test on the three binaries
- `review/a3-subset-{fixed,baseline}.log`, `review/a3-subset-fixed-rerun.log`, `review/run-subset*.sh|.out`
- `review/a3-cases-{fixed,baseline,master}.log` (full run), `review/a3-cases-*-identsplits.log`,
  `review/a3-cases/` (per-case json + server logs, `--verbose`), `review/a3-cases-run1/` (the first run, before the
  expectation fixes), `review/run-cases.sh`, `review/run-one.sh`
- `upstream/llama.cpp-a3-master/build-a3m-{cmake,build}.log`: the master build
- `review/a3_gpu_run.py`, `review/a3_gpu_turn.sh`, `review/a3-fixed-ub512.jsonl`, `review/a3-fixed-ub512-server.log`,
  `review/a3-fixed-ub512-run.log`, `review/a3_gpu_turn.out`: the real-model run (section 4)
- `review/probe-*.log`: model metadata probes (tinygemma3 `n_swa 4096`; LFM2 accepts `--context-shift` and
  `--cache-reuse`)
- `review/dryrun/`: the driver's dry run on the small model; `review/a3_gpu_summarize.py` prints the tables

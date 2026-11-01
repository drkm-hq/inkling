"""Print markdown tables from bench/R1/results/*.json for the report."""
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAYER_MULT = {"0": 2, "1": 33, "2": 7}
LAYER_NAME = {"0": "L0 dense+SWA (x2)", "1": "L1 MoE+SWA (x33)", "2": "L2 MoE+global (x7)"}


def load(tag):
    p = os.path.join(OUT, f"layer_bench_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def decode_table(tag, label):
    r = load(tag)
    if r is None:
        print(f"(missing {tag})")
        return
    print(f"\n#### {label} (`{tag}`): decode, ms per call (median), M5 Max")
    print("| ctx | L0 dense+SWA | L1 MoE+SWA | L2 MoE+global | L1 attn / mlp / sconv | L2 attn / mlp / sconv | 3-layer loop async / sync / build | 42-layer pred. tok/s (parts / loop) | GB/tok | roofline tok/s @480 GB/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ctx, d in sorted(r["decode"].items(), key=lambda kv: int(kv[0])):
        ls = d["layers"]
        e = d["extrapolated_42"]
        f3 = d["full3"]
        print(
            f"| {ctx} | {fmt(ls['0']['layer_ms'])} | {fmt(ls['1']['layer_ms'])} | {fmt(ls['2']['layer_ms'])} "
            f"| {fmt(ls['1']['attn_ms'])} / {fmt(ls['1']['mlp_ms'])} / {fmt(ls['1']['sconv_ms'])} "
            f"| {fmt(ls['2']['attn_ms'])} / {fmt(ls['2']['mlp_ms'])} / {fmt(ls['2']['sconv_ms'])} "
            f"| {fmt(f3['async_ms'],2)} / {fmt(f3['sync_ms'],2)} / {fmt(f3['graph_build_ms'],2)} "
            f"| {fmt(e['tok_s_sum_of_parts'],1)} / {fmt(e['tok_s_loop_based'],1)} | {e['bytes_per_token']/1e9:.2f} | {fmt(e['roofline_tok_s_at_480GBs'],1)} |"
        )
    h = r["head"]
    print(f"\nhead: lm_head {fmt(h['lm_head_ms'])} ms ({h['lm_head_bytes']/1e9:.2f} GB), norm+lm_head+argmax {fmt(h['norm_lmhead_argmax_ms'])} ms, embed {fmt(h['embed_ms'])} ms; residual dtype {r.get('residual_dtype_after_3_layers')}; peak memory {fmt(r.get('peak_memory_gb'), 1)} GB")
    # extras
    print("\nextras (ms): ", end="")
    for ctx, d in sorted(r["decode"].items(), key=lambda kv: int(kv[0])):
        ls = d["layers"]
        ex = {k: v for li in ("1", "2") for k, v in ls[li].items() if k.endswith("_ms") and k not in ("layer_ms", "attn_ms", "mlp_ms", "sconv_ms", "norm_ms")}
        print(f"ctx {ctx}: " + ", ".join(f"{k}={fmt(v)}" for k, v in ex.items()) + "; ", end="")
    print()


def prefill_table(tag, label):
    r = load(tag)
    if r is None or not r.get("prefill"):
        return
    print(f"\n#### {label} (`{tag}`): prefill, ms per layer call (median)")
    print("| T | L0 (attn/mlp/sconv) | L1 (attn/mlp/sconv) | L2 (attn/mlp/sconv) | L2 attn without tau | L1 routed / shared | 42-layer ms | pred. tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for T, p in sorted(r["prefill"].items(), key=lambda kv: int(kv[0])):
        ls = p["layers"]
        e = p["extrapolated_42"]
        def trip(l):
            return f"{fmt(l['layer_ms'],1)} ({fmt(l['attn_ms'],1)}/{fmt(l['mlp_ms'],1)}/{fmt(l['sconv_ms'],1)})"
        print(f"| {T} | {trip(ls['0'])} | {trip(ls['1'])} | {trip(ls['2'])} | {fmt(ls['2'].get('attn_no_tau_ms'),1)} | {fmt(ls['1'].get('routed_experts_ms'),1)} / {fmt(ls['1'].get('shared_experts_ms'),1)} | {fmt(e['ms'],0)} | {fmt(e['tok_s'],0)} |")
        chunks = {k: v for li in ("1", "2") for k, v in ls[li].items() if k.startswith("chunk")}
        if chunks:
            print("  chunk-2048 steady state: " + ", ".join(f"L{li} {k}={fmt(v,1)}" for li in ("1", "2") for k, v in ls[li].items() if k.startswith("chunk")))


if __name__ == "__main__":
    for tag, label in [
        ("e2_abf16_hbf16_decode2", "2-bit tier as shipped (experts q2, attention bf16, lm_head bf16)"),
        ("e2_abf16_hbf16", "2-bit tier as shipped, first clean run (no GPU warm-up)"),
        ("e4_abf16_hbf16_decode2", "4-bit tier as shipped (experts q4, attention bf16, lm_head bf16)"),
        ("e4_abf16_hbf16", "4-bit tier as shipped, first clean run (no GPU warm-up)"),
        ("e2_a8_h8_decode2", "hypothetical: experts q2 + attention q8 + lm_head q8"),
        ("e4_a8_h8", "hypothetical: experts q4 + attention q8 + lm_head q8 (no warm-up)"),
        ("e4_a4_h4_decode2", "hypothetical: everything q4"),
        ("e2_a4_h4", "hypothetical: experts q2 + attention q4 + lm_head q4 (no warm-up)"),
        ("e2_abf16_hbf16_final", "FINAL PASS (contended, canary 8.5 ms): 2-bit tier"),
        ("e4_abf16_hbf16_final", "FINAL PASS (contended, canary 9.0 ms): 4-bit tier"),
        ("e2_a8_h8_final", "FINAL PASS (contended, canary 8.7 ms): q2 + q8 attn + q8 head"),
        ("e4_a8_h8_final", "FINAL PASS (canary 5.7 ms): q4 + q8 attn + q8 head"),
        ("e4_a4_h4_final", "FINAL PASS (quiet, canary 4.6 ms): all q4"),
    ]:
        decode_table(tag, label)
    for tag, label in [("e2_abf16_hbf16", "2-bit tier"), ("e4_abf16_hbf16", "4-bit tier")]:
        prefill_table(tag, label)
    for name in ("kernels_bench", "sdpa_probe", "mtp_probe", "diag_loop", "bw_probe2"):
        p = os.path.join(OUT, name + ".json")
        if os.path.exists(p):
            print(f"\n#### {name}.json")
            d = json.load(open(p))
            if name == "bw_probe2":
                d = d["results"]
                print("| op | ms/op | MB | GB/s |\n|---|---|---|---|")
                for k, v in d.items():
                    print(f"| {k} | {v['ms_per_op']:.4f} | {v['bytes']/1e6:.1f} | {v['GB/s']:.0f} |")
            elif name == "kernels_bench":
                print("| measurement | ms |\n|---|---|")
                for k, v in d.items():
                    if isinstance(v, dict) and "ms" in v:
                        print(f"| {k} | {v['ms']:.4f} {v.get('note','')} |")
            else:
                print("```json\n" + json.dumps(d, indent=1)[:6000] + "\n```")

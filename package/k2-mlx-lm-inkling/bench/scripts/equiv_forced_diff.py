"""Compare a greedy run (the teacher, e.g. mlx-vlm) with a teacher-forced run of another
implementation on the same token stream: at every step both models have seen the identical
prefix, so their top-8 logits are directly comparable.

  python equiv_forced_diff.py ../results/K2-equiv-mlx_vlm.json ../results/K2-equiv-mlx_lm-forced.json
"""
import json
import sys


def main(teacher_path, forced_path):
    t, f = json.load(open(teacher_path)), json.load(open(forced_path))
    print(f"teacher = {t['impl']} {t['version']} ({t.get('python', '?')}) | forced = {f['impl']} {f['version']} ({f.get('python', '?')})")
    for rt, rf in zip(t["runs"], f["runs"]):
        assert rt["name"] == rf["name"] and rt["prompt_tokens"] == rf["prompt_tokens"]
        assert rf["fed"] == rt["tokens"], "the forced run did not consume the teacher's tokens"
        n = len(rt["tokens"])
        agree = [a == b for a, b in zip(rt["tokens"], rf["tokens"])]
        d_top1, d_common, disagreements = [], [], []
        for step, (ta, fa) in enumerate(zip(rt["top8"], rf["top8"])):
            tmap, fmap = dict(ta), dict(fa)
            common = set(tmap) & set(fmap)
            if common:
                d_common.append(max(abs(tmap[k] - fmap[k]) for k in common))
            tok = ta[0][0]
            if tok in fmap:
                d_top1.append(abs(ta[0][1] - fmap[tok]))
            if not agree[step]:
                ftok = fa[0][0]
                disagreements.append(
                    f"step {step}: teacher {tok} (logit {ta[0][1]:.3f}, margin {rt['margins'][step]:.3f}) vs "
                    f"forced argmax {ftok} (logit {fa[0][1]:.3f}, margin {rf['margins'][step]:.3f}); "
                    f"forced logit of teacher's token {fmap.get(tok, float('nan')):.3f}, "
                    f"teacher logit of forced token {tmap.get(ftok, float('nan')):.3f}")
        print(f"\n{rt['name']:20s} prompt {rt['prompt_tokens']:6d}: argmax agreement {sum(agree)}/{n}; "
              f"|dlogit| of the teacher's top-1 token: mean {sum(d_top1) / max(len(d_top1), 1):.4f}, max {max(d_top1) if d_top1 else 0:.4f}; "
              f"max |dlogit| over common top-8 tokens: {max(d_common) if d_common else 0:.4f}")
        for line in disagreements[:12]:
            print("   " + line)
        if len(disagreements) > 12:
            print(f"   ... {len(disagreements) - 12} more")
        if "vs_reference" in rf:
            st = rf["vs_reference"]
            print(f"   logprobs vs reference on the same tokens: KL(ref||mlx-lm) mean {st['kl_mean']:.4f} median {st['kl_median']:.4f} "
                  f"max {st['kl_max']:.4f} nats; |dlogprob| of ref top-1 mean {st['d_top1_mean']:.4f} max {st['d_top1_max']:.4f}; "
                  f"over ref top-100 max {st['d_top100_max']:.4f}; top-1 agreement {100 * st['top1_agreement']:.1f}%")
        if "noise_floor" in rt:
            st = rt["noise_floor"]
            print(f"   noise floor (reference vs itself, prefill chunks {st['chunks']}): KL mean {st['kl_mean']:.4f} median {st['kl_median']:.4f} "
                  f"max {st['kl_max']:.4f} nats; |dlogprob| top-1 mean {st['d_top1_mean']:.4f} max {st['d_top1_max']:.4f}; "
                  f"top-1 agreement {100 * st['top1_agreement']:.1f}%")
        if "router_audit" in rf:
            print("   router audit:", rf["router_audit"]["summary"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

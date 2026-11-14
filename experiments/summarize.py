"""Turns results/objective{3,4}/metrics.json into markdown tables (mean ± std over seeds)."""
import json
import os
import sys
import numpy as np

SHORT = {"sentinel2": "S2", "spot6": "SPOT", "sentinel1_sar": "SAR"}
V4 = [("source_only", "Source-only (S2 model, no adaptation)"),
      ("meanstd", "Test-time mean/std matching to S2"),
      ("histmatch", "Test-time histogram matching to S2"),
      ("gen_test_time", "Test-time generator translation to S2"),
      ("gen_raw_ensemble", "Generator + raw ensemble (old dual-path)"),
      ("train_stats", "Train on mean/std-matched S2 -> target"),
      ("train_gen", "Train on generator-translated S2 -> target"),
      ("train_s2_plus_gen_all", "Train on S2 + generator translations (all targets)"),
      ("uda_dann_entropy", "DANN + entropy UDA (existing recipe)"),
      ("target_supervised", "Target-supervised upper bound (target labels)")]


def ms(vals):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def obj4(path):
    d = json.load(open(path))
    ci = d.get("bootstrap_gain_vs_source_only_ci95_seed0", {})
    M = d.get("metrics", d)
    seeds = sorted(M)
    lines = [f"Objective 4 — held-out 300 patches/sensor, seeds {seeds} (mean ± std over seeds)\n"]
    for t in ["spot6", "sentinel1_sar"]:
        lines.append(f"\n**Target sensor: {SHORT[t]}** (all adaptation uses only unlabeled {SHORT[t]} patches, except the supervised bound)\n")
        lines.append("| Variant | mIoU | Acc | Macro-F1 | mIoU gain vs source-only (seed-0 95% CI) |")
        lines.append("|---|---|---|---|---|")
        base = np.mean([M[s][t]["source_only"]["mIoU"] for s in seeds])
        for k, label in V4:
            vals = [M[s][t][k] for s in seeds if k in M[s][t]]
            if not vals:
                continue
            gain = np.mean([v["mIoU"] for v in vals]) - base
            c = ci.get(t, {}).get(k)
            cis = f" [{c[0]:+.2f}, {c[1]:+.2f}]" if c else ""
            lines.append(f"| {label} | {ms([v['mIoU'] for v in vals])} | {ms([v['acc'] for v in vals])} | "
                         f"{ms([v['f1'] for v in vals])} | {gain:+.2f}{cis} |")
    lines.append("\n**Source sensor S2 (in-domain)**\n")
    lines.append("| Model | mIoU | Acc | Macro-F1 |")
    lines.append("|---|---|---|---|")
    for k in ["source_only", "train_s2_plus_gen_all"]:
        vals = [M[s]["sentinel2"][k] for s in seeds if k in M[s]["sentinel2"]]
        if vals:
            lines.append(f"| {dict(V4)[k]} | {ms([v['mIoU'] for v in vals])} | {ms([v['acc'] for v in vals])} | {ms([v['f1'] for v in vals])} |")
    return "\n".join(lines)


def obj3(path):
    M = json.load(open(path))
    seeds = sorted(M)
    lines = [f"Objective 3 — leave-one-pair-out, seeds {seeds} (mean ± std)\n"]
    pairs = [("sentinel2", "spot6"), ("sentinel2", "sentinel1_sar"), ("spot6", "sentinel1_sar")]
    lines.append("\n**Fidelity of held-out translations vs the aligned real target** (PSNR dB / SSIM / Sobel edge correlation)\n")
    lines.append("| Held-out pair, direction | Generator | PSNR | SSIM | Edge corr |")
    lines.append("|---|---|---|---|---|")
    for a, b in pairs:
        name = f"hold_{a}_{b}"
        for s, t in ((a, b), (b, a)):
            key = f"{s}->{t}"
            for gname, label in (("full", "seen in training (full)"), (name, "**never seen** (held out)")):
                vals = [M[x][gname]["fidelity"][key]["translation"] for x in seeds if gname in M[x]]
                if vals:
                    lines.append(f"| {SHORT[s]}→{SHORT[t]} | {label} | {ms([v['psnr'] for v in vals])} | "
                                 f"{ms([v['ssim'] for v in vals])} | {ms([v['edge_corr'] for v in vals])} |")
            ref = [M[x]["full"]["fidelity"][key] for x in seeds if "full" in M[x]]
            for rk, rl in (("copy_source", "copy source"), ("constant_target_mean", "constant target-mean")):
                lines.append(f"| {SHORT[s]}→{SHORT[t]} | reference: {rl} | {ms([r[rk]['psnr'] for r in ref])} | "
                             f"{ms([r[rk]['ssim'] for r in ref])} | {ms([r[rk]['edge_corr'] for r in ref])} |")
    if any("downstream" in M[s].get("full", {}) for s in seeds):
        lines.append("\n**Downstream mIoU (S2-labeled) when the translation pair is held out**\n")
        lines.append("| Use | Pair | No adaptation / baseline | Full generator | Held-out-pair generator |")
        lines.append("|---|---|---|---|---|")
        for t in ["spot6", "sentinel1_sar"]:
            name = f"hold_sentinel2_{t}"
            base = [M[s]["downstream_source_only"][t]["mIoU"] for s in seeds if "downstream_source_only" in M[s]]
            for use, key in (("test-time target→S2", f"test_time_{t}->s2"), ("train on S2→target", f"train_s2->{t}")):
                f = [M[s]["full"]["downstream"][key]["mIoU"] for s in seeds if key in M[s]["full"].get("downstream", {})]
                h = [M[s][name]["downstream"][key]["mIoU"] for s in seeds if key in M[s].get(name, {}).get("downstream", {})]
                if f and h:
                    lines.append(f"| {use} | S2↔{SHORT[t]} | {ms(base)} | {ms(f)} | {ms(h)} |")
    return "\n".join(lines)


def obj3_paired(path):
    """Per-seed paired difference (held-out-pair generator minus full generator, same seed) in PSNR."""
    M = json.load(open(path))
    seeds = sorted(M)
    lines = [f"\n**Paired difference in PSNR, held-out-pair generator minus full generator (same seed), seeds {seeds}**\n",
             "| Direction | Δ PSNR per seed (dB) | mean ± std | seeds with Δ ≥ 0 |", "|---|---|---|---|"]
    for a, b in [("sentinel2", "spot6"), ("sentinel2", "sentinel1_sar"), ("spot6", "sentinel1_sar")]:
        name = f"hold_{a}_{b}"
        for s, t in ((a, b), (b, a)):
            key = f"{s}->{t}"
            d = [M[x][name]["fidelity"][key]["translation"]["psnr"] - M[x]["full"]["fidelity"][key]["translation"]["psnr"]
                 for x in seeds if name in M[x] and "full" in M[x]]
            if d:
                lines.append(f"| {SHORT[s]}→{SHORT[t]} | {', '.join(f'{v:+.2f}' for v in d)} | {ms(d)} | {sum(v >= 0 for v in d)}/{len(d)} |")
    return "\n".join(lines)


def zero_shot(path):
    """Zero-shot SpaceNet 6 table: every generator, every setting, with 95% bootstrap CIs of the paired differences."""
    R = json.load(open(path))
    gens = [g for g in R if isinstance(R[g], dict)]
    settings = sorted({k for g in gens for k in R[g]})
    fmt = lambda d: (f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
                     if isinstance(d.get("ci95"), list) and len(d["ci95"]) == 2 else f"{d['mean']:+.2f}")
    lines = [f"Zero-shot SpaceNet 6 (n={R.get('n_patches')} patches) — generators never saw SN6; PSNR dB, mean [95% bootstrap CI over patches]\n"]
    for st in settings:
        lines += [f"\n**{st}**\n", "| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |", "|---|---|---|---|---|---|"]
        ref = None
        for g in gens:
            r = R[g].get(st)
            if not r:
                continue
            ref = ref or r
            lines.append(f"| {g} | {r['translation']['psnr']:.2f} | {r['translation']['ssim']:.3f} | {r['translation']['edge_corr']:.3f} | "
                         f"{fmt(r['translation_minus_copy_source']['psnr'])} | {fmt(r['translation_minus_constant_target_mean']['psnr'])} |")
        if ref:
            for rk, rl in (("copy_source", "reference: copy source"), ("constant_target_mean", "reference: constant target-mean")):
                lines.append(f"| {rl} | {ref[rk]['psnr']:.2f} | {ref[rk]['ssim']:.3f} | {ref[rk]['edge_corr']:.3f} | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = []
    p4, p3 = os.path.join(root, "objective4", "metrics.json"), os.path.join(root, "objective3", "metrics.json")
    pz = os.path.join(root, "objective3", "zero_shot_sn6.json")
    if os.path.exists(p4):
        out.append(obj4(p4))
    if os.path.exists(p3):
        out.append(obj3(p3) + "\n" + obj3_paired(p3))
    if os.path.exists(pz):
        out.append(zero_shot(pz))
    text = "\n\n".join(out)
    print(text)
    open(os.path.join(root, "summary.md"), "w", encoding="utf-8").write(text + "\n")

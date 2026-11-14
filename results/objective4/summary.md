Objective 4 — held-out 300 patches/sensor, seeds ['0', '1', '2'] (mean ± std over seeds)


**Target sensor: SPOT** (all adaptation uses only unlabeled SPOT patches, except the supervised bound)

| Variant | mIoU | Acc | Macro-F1 | mIoU gain vs source-only, mean over seeds | gain in seed 0 [95% CI over patches] |
|---|---|---|---|---|---|
| Reference: predict the most frequent training class everywhere | 2.25 | 40.58 | 3.21 | | |
| Source-only (S2 model, no adaptation) | 2.26 ± 0.00 | 40.58 ± 0.00 | 3.21 ± 0.00 | +0.00 | +0.00 |
| Test-time mean/std matching to S2 | 5.08 ± 0.49 | 30.68 ± 2.38 | 7.98 ± 0.77 | +2.82 | +3.50 [+2.98, +3.93] |
| Test-time histogram matching to S2 | 5.09 ± 0.74 | 33.04 ± 4.69 | 8.13 ± 1.05 | +2.84 | +3.80 [+3.33, +4.20] |
| Test-time generator translation to S2 | 1.07 ± 0.87 | 12.26 ± 12.03 | 1.69 ± 1.22 | -1.19 | -1.40 [-1.61, -1.24] |
| Generator + raw ensemble (old dual-path) | 1.57 ± 0.81 | 22.95 ± 14.64 | 2.36 ± 1.09 | -0.69 | -0.34 [-0.45, -0.23] |
| Train on mean/std-matched S2 -> target | 5.69 ± 0.15 | 32.89 ± 0.31 | 8.97 ± 0.27 | +3.43 | +3.61 [+3.10, +4.12] |
| Train on generator-translated S2 -> target | 1.78 ± 0.11 | 17.09 ± 1.37 | 2.81 ± 0.15 | -0.47 | -0.39 [-0.63, -0.25] |
| Train on S2 + generator translations (all targets) | 3.15 ± 0.08 | 38.55 ± 0.38 | 4.64 ± 0.11 | +0.90 | +0.98 [+0.63, +1.25] |
| DANN + entropy UDA (existing recipe) | 2.45 ± 0.70 | 20.71 ± 12.74 | 4.00 ± 0.85 | +0.19 | +1.18 [+0.82, +1.52] |
| Target-supervised upper bound (target labels) | 15.87 ± 0.20 | 60.80 ± 0.13 | 22.57 ± 0.28 | +13.61 | +13.79 [+12.91, +14.39] |

**Target sensor: SAR** (all adaptation uses only unlabeled SAR patches, except the supervised bound)

| Variant | mIoU | Acc | Macro-F1 | mIoU gain vs source-only, mean over seeds | gain in seed 0 [95% CI over patches] |
|---|---|---|---|---|---|
| Reference: predict the most frequent training class everywhere | 2.25 | 40.58 | 3.21 | | |
| Source-only (S2 model, no adaptation) | 0.74 ± 0.90 | 13.06 ± 15.98 | 1.12 ± 1.31 | +0.00 | +0.00 |
| Test-time mean/std matching to S2 | 3.32 ± 0.38 | 34.95 ± 1.51 | 5.16 ± 0.56 | +2.58 | +3.58 [+3.42, +3.91] |
| Test-time histogram matching to S2 | 3.15 ± 0.21 | 32.50 ± 4.53 | 4.55 ± 0.21 | +2.41 | +2.84 [+2.73, +3.11] |
| Test-time generator translation to S2 | 0.78 ± 0.89 | 13.13 ± 15.65 | 1.20 ± 1.28 | +0.04 | +0.10 [+0.07, +0.15] |
| Generator + raw ensemble (old dual-path) | 0.75 ± 0.92 | 13.31 ± 16.30 | 1.13 ± 1.32 | +0.01 | +0.01 [+0.00, +0.01] |
| Train on mean/std-matched S2 -> target | 2.26 ± 0.23 | 32.62 ± 5.05 | 3.53 ± 0.26 | +1.52 | +2.18 [+2.06, +2.31] |
| Train on generator-translated S2 -> target | 1.45 ± 0.15 | 14.13 ± 1.48 | 2.34 ± 0.22 | +0.71 | +1.10 [+0.97, +1.22] |
| Train on S2 + generator translations (all targets) | 1.82 ± 0.26 | 23.42 ± 4.80 | 2.93 ± 0.38 | +1.09 | +1.89 [+1.65, +2.15] |
| DANN + entropy UDA (existing recipe) | 1.59 ± 0.70 | 18.61 ± 8.81 | 2.48 ± 0.90 | +0.86 | +1.53 [+1.42, +1.73] |
| Target-supervised upper bound (target labels) | 19.87 ± 0.07 | 60.45 ± 0.18 | 28.19 ± 0.06 | +19.13 | +19.73 [+18.72, +20.55] |

**Source sensor S2 (in-domain)**

| Model | mIoU | Acc | Macro-F1 |
|---|---|---|---|
| Source-only (S2 model, no adaptation) | 15.75 ± 0.27 | 62.29 ± 0.44 | 23.34 ± 0.39 |
| Train on S2 + generator translations (all targets) | 11.54 ± 0.21 | 59.74 ± 0.96 | 16.80 ± 0.14 |

**How many classes each model predicts on the target (seed 0, 19 land-cover classes + void)** — a model that predicts one class everywhere scores like the most-frequent-class reference

| Variant | SPOT: classes predicted / top-class share | SAR: classes predicted / top-class share |
|---|---|---|
| Source-only (S2 model, no adaptation) | 2 / 99.8% | 2 / 99.9% |
| Test-time mean/std matching to S2 | 19 / 27.5% | 12 / 68.5% |
| Test-time histogram matching to S2 | 19 / 31.0% | 11 / 44.4% |
| Test-time generator translation to S2 | 7 / 67.2% | 6 / 96.2% |
| Generator + raw ensemble (old dual-path) | 4 / 60.6% | 3 / 99.8% |
| Train on mean/std-matched S2 -> target | 17 / 22.4% | 12 / 91.0% |
| Train on generator-translated S2 -> target | 8 / 54.3% | 6 / 64.1% |
| Train on S2 + generator translations (all targets) | 9 / 76.9% | 13 / 70.7% |
| DANN + entropy UDA (existing recipe) | 6 / 85.6% | 7 / 60.3% |
| Target-supervised upper bound (target labels) | 19 / 39.8% | 19 / 38.9% |

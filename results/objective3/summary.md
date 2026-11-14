Objective 3 — leave-one-pair-out, seeds ['0', '1', '2'] (mean ± std)


**Fidelity of held-out translations vs the aligned real target** (PSNR dB / SSIM / Sobel edge correlation)

| Held-out pair, direction | Generator | PSNR | SSIM | Edge corr |
|---|---|---|---|---|
| S2→SPOT | seen in training (full) | 15.81 ± 0.30 | 0.43 ± 0.01 | 0.45 ± 0.01 |
| S2→SPOT | **never seen** (held out) | 16.33 ± 0.30 | 0.43 ± 0.00 | 0.47 ± 0.01 |
| S2→SPOT | reference: copy source | 12.78 ± 0.00 | 0.36 ± 0.00 | 0.58 ± 0.00 |
| S2→SPOT | reference: constant target-mean | 16.98 ± 0.00 | 0.43 ± 0.00 | 0.41 ± 0.00 |
| SPOT→S2 | seen in training (full) | 10.70 ± 2.13 | 0.31 ± 0.03 | 0.43 ± 0.04 |
| SPOT→S2 | **never seen** (held out) | 10.73 ± 0.61 | 0.30 ± 0.01 | 0.39 ± 0.03 |
| SPOT→S2 | reference: copy source | 12.78 ± 0.00 | 0.36 ± 0.00 | 0.58 ± 0.00 |
| SPOT→S2 | reference: constant target-mean | 17.09 ± 0.00 | 0.46 ± 0.00 | 0.74 ± 0.00 |
| S2→SAR | seen in training (full) | 6.72 ± 0.35 | 0.10 ± 0.01 | 0.24 ± 0.02 |
| S2→SAR | **never seen** (held out) | 7.05 ± 0.31 | 0.10 ± 0.00 | 0.25 ± 0.01 |
| S2→SAR | reference: copy source | 5.69 ± 0.00 | 0.10 ± 0.00 | 0.32 ± 0.00 |
| S2→SAR | reference: constant target-mean | 20.35 ± 0.00 | 0.41 ± 0.00 | 0.25 ± 0.00 |
| SAR→S2 | seen in training (full) | 7.52 ± 0.78 | 0.12 ± 0.00 | 0.17 ± 0.05 |
| SAR→S2 | **never seen** (held out) | 8.69 ± 0.73 | 0.12 ± 0.00 | 0.15 ± 0.02 |
| SAR→S2 | reference: copy source | 5.69 ± 0.00 | 0.10 ± 0.00 | 0.32 ± 0.00 |
| SAR→S2 | reference: constant target-mean | 17.09 ± 0.00 | 0.46 ± 0.00 | 0.74 ± 0.00 |
| SPOT→SAR | seen in training (full) | 11.52 ± 1.24 | 0.19 ± 0.02 | 0.18 ± 0.01 |
| SPOT→SAR | **never seen** (held out) | 11.32 ± 0.40 | 0.19 ± 0.01 | 0.18 ± 0.01 |
| SPOT→SAR | reference: copy source | 7.31 ± 0.00 | 0.14 ± 0.00 | 0.24 ± 0.00 |
| SPOT→SAR | reference: constant target-mean | 20.35 ± 0.00 | 0.41 ± 0.00 | 0.25 ± 0.00 |
| SAR→SPOT | seen in training (full) | 15.99 ± 0.32 | 0.23 ± 0.02 | 0.31 ± 0.03 |
| SAR→SPOT | **never seen** (held out) | 15.92 ± 0.25 | 0.24 ± 0.00 | 0.30 ± 0.01 |
| SAR→SPOT | reference: copy source | 7.31 ± 0.00 | 0.14 ± 0.00 | 0.24 ± 0.00 |
| SAR→SPOT | reference: constant target-mean | 16.98 ± 0.00 | 0.43 ± 0.00 | 0.41 ± 0.00 |

**Paired difference in PSNR, held-out-pair generator minus full generator (same seed), seeds ['0', '1', '2']**

| Direction | Δ PSNR per seed (dB) | mean ± std | seeds with Δ ≥ 0 |
|---|---|---|---|
| S2→SPOT | +0.55, +0.51, +0.50 | 0.52 ± 0.02 | 3/3 |
| SPOT→S2 | -2.68, +1.96, +0.80 | 0.03 ± 1.97 | 2/3 |
| S2→SAR | +1.09, +0.30, -0.39 | 0.33 ± 0.60 | 2/3 |
| SAR→S2 | +3.09, +0.60, -0.18 | 1.17 ± 1.40 | 2/3 |
| SPOT→SAR | +0.62, -1.53, +0.30 | -0.21 ± 0.95 | 2/3 |
| SAR→SPOT | +0.01, -0.06, -0.17 | -0.07 ± 0.08 | 1/3 |

Zero-shot SpaceNet 6 (n=2274 patches) — generators never saw SN6; PSNR dB, mean [95% bootstrap CI over patches]


**pastis_heldout sentinel1_sar->spot6 [canonical style]**

| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |
|---|---|---|---|---|---|
| published | 16.65 | 0.259 | 0.377 | +9.29 [+9.04, +9.53] | -0.58 [-0.74, -0.43] |
| full_seed0 | 15.79 | 0.203 | 0.271 | +8.42 [+8.29, +8.56] | -1.45 [-1.51, -1.38] |
| full_seed1 | 16.10 | 0.245 | 0.320 | +8.74 [+8.59, +8.88] | -1.13 [-1.19, -1.07] |
| full_seed2 | 16.66 | 0.247 | 0.336 | +9.30 [+9.11, +9.49] | -0.57 [-0.65, -0.49] |
| reference: copy source | 7.36 | 0.144 | 0.238 | | |
| reference: constant target-mean | 17.23 | 0.432 | 0.406 | | |

**pastis_heldout spot6->sentinel1_sar [canonical style]**

| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |
|---|---|---|---|---|---|
| published | 16.29 | 0.340 | 0.183 | +8.93 [+8.78, +9.06] | -4.16 [-4.33, -3.99] |
| full_seed0 | 10.90 | 0.181 | 0.174 | +3.54 [+3.44, +3.63] | -9.55 [-9.69, -9.41] |
| full_seed1 | 13.41 | 0.224 | 0.173 | +6.05 [+5.88, +6.22] | -7.04 [-7.21, -6.87] |
| full_seed2 | 10.59 | 0.168 | 0.184 | +3.23 [+3.10, +3.36] | -9.86 [-10.02, -9.69] |
| reference: copy source | 7.36 | 0.144 | 0.238 | | |
| reference: constant target-mean | 20.45 | 0.413 | 0.255 | | |

**sn6 sentinel1_sar->spot6 [canonical style]**

| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |
|---|---|---|---|---|---|
| published | 15.41 | 0.274 | 0.287 | +6.96 [+6.80, +7.12] | -1.15 [-1.21, -1.07] |
| full_seed0 | 15.71 | 0.252 | 0.243 | +7.26 [+7.14, +7.39] | -0.84 [-0.87, -0.81] |
| full_seed1 | 15.89 | 0.274 | 0.261 | +7.44 [+7.32, +7.56] | -0.66 [-0.69, -0.63] |
| full_seed2 | 15.78 | 0.268 | 0.264 | +7.33 [+7.19, +7.47] | -0.77 [-0.82, -0.73] |
| reference: copy source | 8.45 | 0.252 | 0.262 | | |
| reference: constant target-mean | 16.56 | 0.353 | 0.264 | | |

**sn6 sentinel1_sar->spot6 [exemplar style]**

| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |
|---|---|---|---|---|---|
| published | 11.67 | 0.193 | 0.168 | +3.21 [+3.16, +3.26] | -4.89 [-4.98, -4.80] |
| full_seed0 | 8.57 | 0.202 | 0.187 | +0.12 [+0.10, +0.14] | -7.98 [-8.11, -7.87] |
| full_seed1 | 8.77 | 0.208 | 0.187 | +0.32 [+0.29, +0.34] | -7.79 [-7.90, -7.68] |
| full_seed2 | 9.57 | 0.198 | 0.183 | +1.11 [+1.04, +1.19] | -6.99 [-7.11, -6.86] |
| reference: copy source | 8.45 | 0.252 | 0.262 | | |
| reference: constant target-mean | 16.56 | 0.353 | 0.264 | | |

**sn6 spot6->sentinel1_sar [canonical style]**

| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |
|---|---|---|---|---|---|
| published | 17.67 | 0.474 | 0.277 | +9.22 [+9.12, +9.33] | -4.89 [-5.00, -4.78] |
| full_seed0 | 11.74 | 0.302 | 0.185 | +3.29 [+3.21, +3.38] | -10.82 [-10.97, -10.67] |
| full_seed1 | 14.53 | 0.367 | 0.229 | +6.08 [+5.97, +6.19] | -8.03 [-8.17, -7.89] |
| full_seed2 | 11.45 | 0.281 | 0.208 | +3.00 [+2.91, +3.09] | -11.11 [-11.25, -10.96] |
| reference: copy source | 8.45 | 0.252 | 0.262 | | |
| reference: constant target-mean | 22.56 | 0.666 | 0.557 | | |

**sn6 spot6->sentinel1_sar [exemplar style]**

| Generator | PSNR | SSIM | Edge corr | Δ PSNR vs copy-source | Δ PSNR vs constant target-mean |
|---|---|---|---|---|---|
| published | 15.05 | 0.375 | 0.214 | +6.60 [+6.47, +6.74] | -7.51 [-7.66, -7.36] |
| full_seed0 | 9.10 | 0.267 | 0.207 | +0.65 [+0.57, +0.72] | -13.46 [-13.61, -13.32] |
| full_seed1 | 10.45 | 0.299 | 0.216 | +2.00 [+1.88, +2.13] | -12.11 [-12.28, -11.93] |
| full_seed2 | 9.61 | 0.262 | 0.207 | +1.16 [+1.07, +1.25] | -12.95 [-13.10, -12.79] |
| reference: copy source | 8.45 | 0.252 | 0.262 | | |
| reference: constant target-mean | 22.56 | 0.666 | 0.557 | | |

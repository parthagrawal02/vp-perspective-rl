# FINAL perspective reward (frozen 2026-11-17, Task 11)

**Frozen.** Do not change the definition, parameters or constants without explicit approval (Sagnik / coordinator).
Decision record: α 0.8, λ_f 4, Q = PickScore + aesthetic, λ_q 0.25, contrast stretch. The Task 10 floor-activity cap (20%)
was relaxed to 23% **after seeing the Task 10 results** (this setting has 23.0% floor activity on SELECT base twins).

Code: `vp_reward/vp_reward/final.py` (constants at the top of the file). Geometry helpers: `vp_reward/vp_reward/v4.py`
(`stretch`, `refine_both`, `sth_fixed`, `compute_t8`), `v2.py` (`fit_z`, `a1_score`, `z_of`), `core.py` (`make_segs`, `ang_err`, `refine_vp`).

## 1. Definition

```
R = z(G) − 4·P + 0.25·zQ
z(G) = (G − 0.6762756750467267) / 0.12087393883907192
P    = max(0, 1 − Lc / (0.8 · Lc_base))            (P = 0 if Lc_base ≤ 0)
zQ   = (z(PickScore) + z(aesthetic) − 5.684341886080802e-16) / 1.5849401901583253
z(PickScore) = (PickScore − 22.55044470282162) / 0.989595540429443     PickScore = flow_grpo PickScoreScorer output × 26
z(aesthetic) = (aesthetic − 5.452365678899429) / 0.3416999366620726    aesthetic = flow_grpo AestheticScorer output
```
Higher R = better. Images: 512×512 RGB uint8 (the entry point asserts this size).

## 2. Geometry score G (p2 sig) and Lc

1. **Preprocessing:** RGB → grayscale (OpenCV `COLOR_RGB2GRAY`), float32; map the 1st and 99th intensity percentiles to 0 and 255
   (`(g − p1) · 255 / max(p99 − p1, 1e-3)`), clip to [0, 255]. **No blur.** Applied after any degradation (same order as in RL).
2. **Lines:** DeepLSD, wireframe weights (`models/deeplsd/deeplsd_wireframe.tar`), input = stretched grey / 255,
   `line_detection_params = {merge: False, filtering: True, grad_thresh: 3, grad_nfa: True}`. **No line refiner, no VP optimisation.**
3. **Line refit** (on the stretched image; `v4.refine_both`): segments shorter than 0.9·l_min are skipped; per segment, pixels with
   t ∈ [5%, 95%] of the length and |normal offset| ≤ 2.5 px whose Sobel gradient is within 22.5° of the normal and has magnitude
   > 2σ_g (σ_g = image noise σ·√12/8, `v2.noise_sigma`) are binned by t (1 px bins); if ≥ 10 such pixels and ≥ 5 bins, the line is the
   **unweighted** least-squares fit to the per-bin gradient-weighted centroids; otherwise the DeepLSD endpoints are kept.
   Segments are kept if the V0 refit length is ≥ l_min = 0.04·512 = 20.48 px.
4. **Orientation uncertainty (V2, fixed, no χ²):** σ_θ = σ0·√(12 / (n³ − n)), n = max(0.9·L, 3), **σ0 = 1 px**.
5. **z-score of a segment w.r.t. a VP:** z = (L/2)·|sin α| / √((L/2)²·σ_θ² + σ_sys²), α = angle between the segment and the
   midpoint→VP direction, **σ_sys = 0.1 px** (512 frame).
6. **Sequential RANSAC** (`v2.fit_z`, RNG `numpy.default_rng(0)`): up to **3 VPs**; for each, **512** pair samples drawn with
   probability ∝ segment length from the remaining segments (pairs with i = j dropped), hypothesis = cross product of the two lines
   (normalized coords, image spans [−1, 1]); best hypothesis = max total length with **z < 3**; then 3 rounds of
   {inliers = z < 3 → IRLS refit (`core.refine_vp`, 4 iterations, weights (1/σ_θ²)/|u|²)}.
   **Stop rule:** stop when < 2 segments remain, no valid hypothesis, or the final inlier length < **10%** of the total segment length
   (that VP is discarded); otherwise accept the VP and remove its inliers (z < 3).
7. **Assignment:** each segment → VP with the smallest angular error; assigned if that error < **15°**.
8. **G** = 1 − Σᵢ wᵢ·v(zᵢ) / Σᵢ wᵢ over assigned segments, wᵢ = (Lᵢ/100)², v(z) = sigmoid((z − 3)/0.5) (argument clipped to ±50).
   **Lc** = Σ Lᵢ over assigned segments (px, 512 frame; 0 if G is undefined).
9. **NaN rule:** G is undefined if < 2 kept segments, RANSAC finds no VP, or no assigned length. Then **G := 0.37950405006078053**
   (1st percentile of the 6,797 defined G values of the SELECT base twins; 3 of 6,800 were NaN), and `info["G_nan"] = True`.
   Lc is then 0, so P = 1 (maximal floor penalty).

## 3. Floor and Lc_base

- P = max(0, 1 − Lc / (0.8·Lc_base)).
- **Lc_base** = mean Lc of base SD3.5-M samples for the same prompt.
  In validation (Tasks 9–11) Lc_base = mean Lc of the caption's 4 base twins (CFG 4.5, 40 steps, 512, seeds 0–3), excluding the image
  itself when it is one of those twins (leave-one-out); for all other images, the mean of all 4.
- **OPEN:** which sampler settings (CFG, steps, number of samples) define Lc_base during RL — to be decided in the RL task. Lc_base must
  be precomputed per training prompt and passed in (`lc_base` dict prompt → value, or `metadata[k]["lc_base"]`).

## 4. Standardization constants (Task 10; SELECT base twins of NYU v2, SUN RGB-D kv2, Hypersim; RANSAC seed 0; n = 6,800)

| quantity | mean | std |
|---|---|---|
| G (p2 sig, stretched) | 0.6762756750467267 | 0.12087393883907192 |
| PickScore (×26) | 22.55044470282162 | 0.989595540429443 |
| aesthetic | 5.452365678899429 | 0.3416999366620726 |
| z(PickScore) + z(aesthetic) | 5.684341886080802e-16 | 1.5849401901583253 |

## 5. Code entry point

```python
# env from envs/setup_nft_vp.sh (vp_reward + deeplsd installed); flow_grpo via PYTHONPATH=$VP_NFT_DIR (config/env.sh)
from vp_reward.final import perspective_reward
fn = perspective_reward(device="cuda", lc_base={prompt: Lc_base, ...}, n_workers=16)
scores, info = fn(images, prompts, metadata)   # DiffusionNFT signature; images: NCHW float [0,1] tensor or NHWC uint8
# scores: float32 (B,) = R
# info: G (NaN-filled), G_raw, G_nan (flag), zG, Lc, Lc_base, P, pickscore, aesthetic, zQ, R
```
Building blocks: `geometry_batch(net, ims)` → (G, Lc) needs DeepLSD; `quality_batch(ps, ae, ims, prompts)` → (PickScore, aesthetic)
needs flow_grpo; `combine(G, Lc, Lc_base, pick, aes)` → (R, info) is a pure function.

**Environment note:** at freeze time DeepLSD's dependencies were only in the `deeplsd` env and flow_grpo only in `nft`; since Task 12
both are in `nft-vp` (`envs/setup_nft_vp.sh`), where the reproduction check gives identical G and Lc. Command-line scoring:
`python -m vp_reward.score` (`vp_reward/score.py`, a wrapper that changes no numbers). DeepLSD weights path: `$VP_DEEPLSD_CKPT`.

## 6. Reproduction check

`experiments/task11/repro_final.py`, now `vp_reward/tests/test_repro.py` (sample: 20 random SELECT base twins from Task 10, plus 1 twin with undefined G). Geometry run through
`geometry_batch` (deeplsd env), quality through `quality_batch` (nft env), combined with `combine` and the Task 10 leave-one-out Lc_base:
max |diff| vs Task 10 over the 20 images: **G 0, Lc 0, PickScore 7.6e-6, aesthetic 9.5e-7, R 1.2e-6** (float32 scorer noise).
The NaN twin (hypersim 421 s0) gets G_raw = NaN, flag True, G = 0.3795. Output: `outputs/task11/repro_check.csv`.

## 7. Validation

- SELECT (Task 10): margin, rotation, learnability 7.86, Spearman(R, G) 1.0, floor activity 23.0%.
- Fresh Hypersim split (Task 11; 596 images, `datasets/hypersim_fresh`): G passes all Hypersim criteria (untied wr 0.688, p 1e-34;
  5/5 quintiles; rotation rule; learnability 6.06; degraded-twin wr 0.674). **R fails the exploit margin**: noise σ=2 raises R
  (+0.043, p = 7e-4; both z(G) +0.034 and 0.25·zQ +0.031 rise) and blur 0.5 has mean ΔR +0.014 (n.s.). Floor activity on base twins
  23.8% (> 23%). R learnability 6.93, Spearman(R, G) 1.0. Reported, not tuned. Files: `outputs/task11/{confirm11,exploit11,baseline11,paired11}.csv`.

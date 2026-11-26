# task15: Probe the frozen reward's weak spots; experimental v2 (2026-11-26)

Code: `eval15.py` (stages `fit`, `synth`, `densify`, `report`), `vp_reward/vp_reward/reward_v2.py`, tests `vp_reward/tests/test_v2.py`.
Data: `vp_pairs_v1` JPEG q95 copies (stored metadata G was scored on PNGs, so **every variant here, including the frozen one, is
re-scored on the same JPEG pixels**). Sample: 80 SELECT pairs each of hypersim_select / nyu / sun + 120 hypersim_fresh pairs
(1,800 images, `random.Random(0)`). DeepLSD on Apple MPS (same G as CPU). BETA = 0 reproduces `final.geometry` exactly.
Outputs `$VP_DATA/outputs/task15/` (`fits.pkl`, `discrimination15.csv`, `synth15.csv`, `densify15.csv`, `g_stats.pkl`).

## 1. Monotone line penalty (BETA): rejected
Idea: frozen G ignores segments > 15° from every VP, so a very wrong line costs nothing. v2 option: count unassigned segments
as saturated violators with weight BETA·L².
- Synthetic bend sweep (DeepLSD on rendered scenes): frozen G bottoms out at 8-16° and recovers by 45° (0.821 → 0.728 → 0.765).
  BETA barely changes this (BETA = 1: 0.727 → 0.737). Diagnosis: as the bend grows, fewer bent segments survive detection
  (17 → 8 matched segments at 45°; rotated segments cross others and are split below l_min), so the recovery is mostly
  detection loss, not the gate.
- Real data, untied reference-vs-twin win rate: SELECT 0.621 / 0.612 / 0.592 / 0.569 for BETA 0 / 0.25 / 0.5 / 1;
  fresh 0.729 / 0.713 / 0.721 / 0.694; SUN 0.538 → 0.484 (BETA 0.5) → 0.438. Real photos have many legitimate non-VP lines.
→ v2 keeps BETA = 0 (frozen G). Constants for other BETAs are recorded in `reward_v2.G_STATS` (JPEG, provisional).

## 2. Split vanishing point: not fixed
One structure whose receding lines should share a VP, right half aimed at a displaced VP: G 0.658 (consistent) vs 0.772 /
0.725 / 0.792 (displaced 40 / 120 / 250 px), identical for every BETA. The free third VP absorbs the inconsistency. Fixing it
needs a camera-consistency term (orthogonality given a focal length, horizon), not attempted here.

## 3. Densification and the upper cap (GAMMA = 1.75, LAMBDA_HI = 4)
`densify`: K dark anti-aliased segments drawn exactly toward each twin's own fitted VPs (60 twins, Lc_base = mean Lc of the
other 3 twins, same JPEG pipeline; zQ held at 0).

| K | median Lc/Lc_base | mean ΔG | mean ΔP | ΔR frozen | ΔR v2 | cap active |
|---|---|---|---|---|---|---|
| 20 | 1.07 | −0.002 | −0.033 | +0.114 | +0.104 | 3% |
| 60 | 1.26 | −0.014 | −0.047 | +0.076 | +0.004 | 8% |
| 150 | 1.59 | −0.002 | −0.050 | +0.185 | −0.190 | 42% |

G does not reward densification; the gain is the floor. The 17 twins under the floor at K = 0 gain +0.41 / +0.55 / +0.94 R
(frozen) and +0.41 / +0.55 / +0.77 (v2). The cap turns heavy densification negative on average but cannot remove the
floor-escape incentive, which is the floor working as designed (it asks for structure and cannot tell real from added lines).
Cap calibration: GAMMA = 1.75 is active on 2.9% of base twins / 1.0% of references in vp_pairs_v1 (metadata Lc, LOO Lc_base).
Caveat for RL: if Lc_base comes from the CFG 1.0 training sampler (less structure), the cap can fire on legitimate
CFG-4.5-like structure, so pair v2 with `rl/lcbase.py --sampler validated`.

## Takeaways
- v2 = frozen G + upper cap. A mild guard against unbounded densification; not a fix for the split-VP or noise loopholes.
- The bend-sweep non-monotonicity found earlier is mostly a detection effect in the synthetic scene; the 15° gate is minor.

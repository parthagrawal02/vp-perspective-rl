# vp-perspective-rl

CMU 10-423/623 course project. We fine-tune **Stable Diffusion 3.5 Medium** with online RL (**DiffusionNFT**) using a frozen,
non-learned **perspective-consistency reward**: line segments detected by DeepLSD should agree with up to three fitted
vanishing points. The reward was selected and validated against real indoor photos (NYU Depth v2, SUN RGB-D, Hypersim) and their
caption-matched SD3.5-M "twins" before any training. This repo holds the reward, the validation metrics, the DiffusionNFT
integration (as a patch), the data scripts, and every experiment script as it was run.

**Status (2026-11-24):** reward frozen (`vp_reward/FINAL.md`); RL integration smoke-tested only and **provisional** (see Known issues).

## Quickstart: score a folder of images

```bash
git clone <this repo> vp-perspective-rl && cd vp-perspective-rl
cp config/env.example.sh config/env.sh      # set VP_DATA to a big disk (datasets, outputs, weights, envs, caches go there)
source config/env.sh
bash envs/setup_nft_vp.sh                   # 3 min with a warm pip cache (downloads ~4 GB otherwise): env + DeepLSD + patched DiffusionNFT + vp_reward + DeepLSD/aesthetic weights
source config/env.sh                        # puts the new env on PATH

python -m vp_reward.score path/to/images/                                   # G, zG, Lc for every 512x512 image
python -m vp_reward.score img.png --prompt "A photo of a kitchen ..." --lc-base 11438   # + PickScore, aesthetic, zQ, P, R
python -m vp_reward.score big.jpg --resize                                  # center-crop + resize to 512 first
```
From Python: `from vp_reward.score import score_images; df = score_images(paths, prompts=None, lc_base=None)`; the DiffusionNFT-style
batch reward is `vp_reward.final.perspective_reward(device, lc_base=...)`. A GPU is recommended (DeepLSD); the geometry fit is CPU.
PickScore is prompt-conditional, so the quality term (and R) needs a prompt; the floor P needs `Lc_base`, the mean Lc of base
SD3.5-M samples of the same prompt. Requirements and Blackwell notes: [envs/README.md](envs/README.md).

## The reward in brief

Full spec: **[vp_reward/FINAL.md](vp_reward/FINAL.md)** (frozen 2026-11-17; code `vp_reward/vp_reward/final.py`).

```
R  = z(G) − 4·P + 0.25·zQ
G  : grey → 1–99th-percentile contrast stretch → DeepLSD (wireframe) → keep segments ≥ 4% of 512 px → per-segment refit →
     sequential RANSAC, ≤ 3 VPs → assign each segment to its best VP within 15° → z = angular error / σ(length) (σ0 = 1 px)
     G = 1 − Σ w·sigmoid((z − 3)/0.5) / Σ w,  w = L²        (share of long-line length that is VP-consistent; 1 = perfect)
Lc : total assigned segment length (px). Undefined G (too few lines / no VP) → G := 0.3795, Lc = 0.
P  = max(0, 1 − Lc / (0.8·Lc_base))    structure floor: blurring / erasing lines cannot raise R
zQ = standardized PickScore + aesthetic (guards against image-quality collapse)
```
Images are scored at 512×512. Constants are standardisation statistics of 6,800 SELECT-split base twins (Task 10).

## Repo map

| path | what |
|---|---|
| `vp_reward/` | **Core.** The frozen reward package (`final.py` + its geometry code `v4.py`, `v2.py`, `core.py`), `score.py` (function + CLI), `paths.py` (all paths from env vars), `FINAL.md` (spec), `tests/` (unit tests + 20-image reproduction check). `api.py` / `synth.py`: the Task 2 LSD reward and synthetic test scenes. |
| `eval/` | **Core.** `metrics.py`: paired win rate (ties ½), untied win rate, Lc-quintile win rates, rotation test, degraded-twin test, exploit test, learnability, within-caption Spearman, floor activity. `fresh_split.py` reproduces the Task 11 headline. |
| `rl/` | **Core, provisional.** `diffusionnft_vp.patch` against NVlabs/DiffusionNFT @ `cbb14f8` (adds the `vp_perspective` reward, the `sd3_vp` config, per-epoch reward-part logging), `apply_patch.sh`, `make_dataset.py` (vp_hypersim prompts), `lcbase.py` (Lc_base per prompt), `run_train.sh`. |
| `data/` | **Core.** Reference-set prep (NYU, SUN RGB-D kv2, Hypersim + fresh + train splits), captioning (VLM prompt, 66-token cap), twin generation. Pointer to the packaged dataset `vp_pairs_v1`. |
| `envs/` | Environment specs (`requirements-*.lock.txt`), `setup_nft_vp.sh`, `setup_vlm.sh`, notes (Blackwell / CUDA 12.8, pytlsd build). |
| `config/` | `env.example.sh`: the single configuration (copy to `config/env.sh`, git-ignored). |
| `experiments/taskNN/` | **Archive.** Every task's scripts exactly as run (paths/imports lightly fixed), each with `NOTES.md` (what it did, where outputs went). Task 1 = env + baseline replication, 2–10 = reward design/validation, 11 = freeze, 12 = RL wiring, 13 = report figures, 14 = dataset package. |
| `third_party/` | (git-ignored) DeepLSD and DiffusionNFT clones made by the setup scripts. |

## How to reproduce

Outputs live under `$VP_DATA/outputs/` (not in the repo). "Figure X" = the midway-report figure files of Task 13
(`$VP_DATA/outputs/report_figs/<X>_*.jpg`, each with a `.txt` caption holding its selection rule and numbers).

| report item | script | output |
|---|---|---|
| Fig. A1 / A2: caption-matched pairs (typical / largest gap) | `experiments/task13/make_figs.py` | `report_figs/A1_pairs_typical.jpg`, `A2_pairs_largest_gap.jpg` |
| Fig. B: what the reward sees (overlay) | `experiments/task13/make_figs.py` (`figlib.py`) | `report_figs/B_what_reward_sees.jpg` |
| Fig. C: rotation test example | `experiments/task13/make_figs.py` | `report_figs/C_rotation_test.jpg` |
| Fig. D: why blur fools a line-fraction score | `experiments/task13/make_figs.py` | `report_figs/D_blur_fools_fraction.jpg` |
| Fig. E1 / E2: baselines; Flow-GRPO's largest G drops + long/short analysis | `experiments/task13/make_figs.py` | `report_figs/E1_*.jpg`, `E2_*.jpg`, `e2_long_short.json`, `e2_pairs.csv` |
| Fig. F / G: within-prompt variation; RL smoke-test samples | `experiments/task13/make_figs.py` | `report_figs/F_*.jpg`, `G_*.jpg` |
| Table: validation of G / R on the fresh Hypersim split (win rate 0.688, quintiles, rotation, learnability, degraded twins, exploit margin, floor activity) | `experiments/task11/eval11.py`; headline re-check `python -m eval.fresh_split` | `outputs/task11/confirm11.csv`, `subsets11.csv`, `exploit11.csv` |
| Table: baselines (base CFG 4.5 / 1.0, perspective suffix, best-of-4, Flow-GRPO, DiffusionNFT) + paired win rates vs base | `experiments/task11/eval11.py` (images: `task11/gen11.py`) | `outputs/task11/baseline11.csv`, `paired11.csv` |
| Reward-design history (variants tried, why each failed) | `experiments/task03`–`task10` (`evalv.py`, `eval4.py` … `eval10.py`) | `outputs/task3`–`task10/*.csv` (see each `NOTES.md`) |
| Standardisation constants (FINAL.md §4) | `experiments/task10/pipe10.py`, `eval10.py` | `outputs/task10/table10_data.pkl` |
| Reward reproduction check (20 images) | `pytest vp_reward/tests/test_repro.py -s` (was `task11/repro_final.py`) | printed; `outputs/task11/repro_check.csv` |
| RL smoke test (3 epochs, timings, memory, reward parts) | `experiments/task12/run_smoke.sh` (now `rl/run_train.sh`), `check12.py` | `outputs/task12/nft_logs/*/vp_epoch_stats.csv`, `vp_eval_stats.csv` |
| DrawBench replication of the published baselines | `experiments/task01/run_eval3.sh` | `outputs/eval_drawbench/` |
| Dataset `vp_pairs_v1` (counts, checks) | `experiments/task14/build.py`, `check.py` | `outputs/task14/vp_pairs_v1.zip`, `checks14.json` |

## Data sources and licenses

| source | license | use here |
|---|---|---|
| Hypersim (apple/ml-hypersim) | dataset **CC BY-SA 3.0** (code: Apple sample-code license) | reference images, fresh split, RL prompts |
| NYU Depth v2 (labeled) | **no license stated**; the page asks to cite Silberman et al., ECCV 2012 | reference images |
| SUN RGB-D (kv2 subset) | **no license stated**; cite Song et al., CVPR 2015 and the source datasets (NYU v2, B3DO, SUN3D) | reference images |
| SD3.5-Medium outputs | Stability AI Community License | twins, baselines |
| DiffusionNFT (NVlabs) | Apache-2.0 | RL code; we distribute only a patch, see below |
| DeepLSD (cvg) code + weights | MIT | line detector (cloned, not vendored) |

No data, images, weights or logs are in this repo. The caption-matched pair dataset is packaged separately as `vp_pairs_v1.zip`
(core: 1.71 GB, 20,140 images) + `vp_pairs_v1_baselines.zip` (443 MB); on the project machine they are at
`$VP_DATA/outputs/task14/`; ask the team for a copy. Its README repeats these licenses; because NYU and SUN RGB-D state no license,
it is for private sharing within the course project only.

**DiffusionNFT patch (Apache-2.0).** Apache-2.0 allows us to reproduce and distribute modified versions of the code, including this
patch (which contains upstream context lines), as long as we (a) give recipients a copy of the license
(`rl/LICENSE-DiffusionNFT-Apache-2.0.txt`), (b) state that we changed the files (the patch itself and `rl/README.md` list every
change), and (c) keep upstream copyright notices (the patch does not touch them; DiffusionNFT has no NOTICE file). It grants no
trademark rights and comes without warranty. Our own code here carries no license yet (private repo).

## Known issues / open items

- **Floor-reference mismatch (RL is provisional).** In validation, `Lc_base` is the mean Lc of the caption's 4 base twins sampled at
  CFG 4.5 / 40 steps (leave-one-out). The RL map `lc_base.json` (`rl/lcbase.py`) uses the training-rollout sampler of `sd3_vp`
  (DiffusionNFT's SD3 defaults: 10 dpm2 steps, deterministic, **CFG 1.0**), so the floor P during training is not calibrated like the validated one
  (FINAL.md §3, OPEN). The floor reference is being fixed; do not treat `rl/` results as final.
- **"More lines" (densification) check.** Not yet tested whether the policy can raise G or R by adding extra line structure
  (e.g. dense tiling or grid patterns). The Task 12 smoke-test grid at epoch 2 already looked washed out and grid-heavy.
- **Light-noise loophole.** On the fresh split, Gaussian noise σ = 2 raises R (mean +0.043, p = 7e-4: z(G) +0.034 and 0.25·zQ +0.031)
  and blur 0.5 gives +0.014 (n.s.): **R fails the exploit margin** (`outputs/task11/exploit11.csv`). Reported, not tuned.
- **JPEG re-scoring shifts G.** All stored scores are on lossless PNGs. Re-scoring the dataset's JPEG-q95 copies changes G by
  median 0.018, 90th pct 0.086, max 0.198 (200 images; mean +0.006). Score PNGs when comparing to stored values.
- Floor activity on fresh-split base twins is 23.8% (agreed cap 23%, itself relaxed from 20% after seeing Task 10; FINAL.md).
- The reward is defined at 512×512 only; `--resize` changes the image.

## Checks (Task 15)

Fresh env built from `envs/` exactly as above (not reusing the project env); on it: unit tests pass (`pytest vp_reward -m "not gpu"`),
the 20-image reproduction gives G and Lc identical to the stored values (R within 1.2e-6, float32 scorer noise, as in Task 11),
the CLI scores an image, and `python -m eval.fresh_split` reproduces G win rate 0.688 and quintiles 0.57/0.74/0.70/0.82/0.73.

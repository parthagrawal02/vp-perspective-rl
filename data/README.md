# data/: reference sets, captions, caption-matched twins

Everything is written under `$VP_DATA/datasets/<ds>/` (`rgb512/<id>.png`, `meta.json` with the split, `captions.json`) and
`$VP_DATA/outputs/`. These are the scripts that built the project data, with paths from `config/env.sh` (originals in
`experiments/`, see the `Origin:` line in each docstring).

**Packaged dataset.** The finished pairs are packaged as `vp_pairs_v1.zip` (every reference + its 4 base twins, CFG 4.5, 40 steps,
seeds 0–3; 20,140 JPEG-q95 images; `metadata.csv` with G / Lc per image; `captions.jsonl`; checksums) and
`vp_pairs_v1_baselines.zip` (fresh-split suffix / CFG 1.0 / Flow-GRPO / DiffusionNFT, seeds 0–1). They're not in git; on the
project machine they are at `$VP_DATA/outputs/task14/`, so ask the team for a copy. Built by `experiments/task14/`.

| set (`ds`) | source | prep | images | split |
|---|---|---|---|---|
| `nyu` | NYU Depth v2 labeled `.mat` (download manually from the NYU page to `datasets/nyu/raw/`) | `prep_nyu.py`: 8 px border crop, center square, 512 | 1,449 | scene-level SELECT 705 / REPORT 744 (232 + 232 scenes) |
| `sun_kv2` | SUN RGB-D zip (manual download, extract `kv2/`) | `prep_sun_hypersim.py`: 6 px crop | 985 | session-level SELECT 498 / REPORT 487 |
| `hypersim` | apple/ml-hypersim (clone it to `datasets/hypersim/ml-hypersim`) | `hypersim_dl.py`, then `prep_sun_hypersim.py` | 998 | scene-level SELECT 497 / REPORT 501 (222 + 222 scenes) |
| `hypersim_fresh` | never-used Hypersim frames | `hypersim_fresh_dl.py`, `prep_hypersim_fresh.py` | 596 | held-out validation (Task 11) |
| `hypersim_train` | never-used frames of SELECT scenes | `hypersim_train_dl.py`, `prep_hypersim_train.py` | 1,921 | RL training prompts (Task 12) |

Counts are captioned images (references with no valid caption are dropped: 2 Hypersim, 4 fresh, 6 train). `hypersim_fresh_dl.py`
and `hypersim_train_dl.py` record their frozen selections in `selection.json`; the selection depends on what was already on disk,
so reuse those files rather than re-running it.

**Captions** (`caption.py`, run via `run_caption.sh <ds>` in the vlm env): Qwen/Qwen3.8-27B-FP8 on vLLM 0.33.0, thinking off,
T 0.7, top-p 0.8, top-k 20, presence penalty 1.5. The prompt asks for one "A photo of ..." caption of ≤ 45 words covering room
type, objects/materials/colors, layout, camera viewpoint and lighting, and bans quality / blur / perspective / geometry words. The
**66-token cap** is 75 CLIP tokens minus the 9 tokens of the perspective suffix `", architectural photography, straight lines,
correct perspective"`, counted as the max over SD3.5's two CLIP tokenizers without special tokens. Invalid captions are
shortened by the VLM, then re-captioned (≤ 5 rounds).

**Twins** (`gen_twins.py`): SD3.5-Medium, 512², 40 steps, CFG 4.5, seeds 0–3 (one CPU `torch.Generator` per sample), fp16
transformer + text encoders, fp32 VAE, no prompt suffix. Baselines on the fresh split (suffix, CFG 1.0, Flow-GRPO LoRA
`jieliu/SD3.5M-FlowGRPO-PickScore`, DiffusionNFT LoRA `worstcoder/SD3.5M-DiffusionNFT-MultiReward`) are in
`experiments/task11/gen11.py`. Degraded twins and rotated references (for the validation tests) are made in
`experiments/task03/pipe.py` / `imageops.py` (Tasks 2–3; degradation tuned per set to the reference sharpness).

Licenses: Hypersim CC BY-SA 3.0; NYU v2 and SUN RGB-D have no stated license (cite them); see the top-level README.

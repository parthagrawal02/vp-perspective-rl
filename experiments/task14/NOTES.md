# task14: Package the caption-matched pairs dataset vp_pairs_v1 (2026-11-20)

`build.py` ran DeepLSD on the 8,660 REPORT-side images without cached lines (+40 determinism re-checks), scored every image with the frozen reward, and exported JPEG-q95 images, metadata.csv, captions.jsonl and checksums; `check.py` verified rows/files/checksums/counts and that 16,248 stored scores match exactly; `readme.py` wrote the dataset READMEs. Outputs `$VP_DATA/outputs/task14/vp_pairs_v1.zip` (1.71 GB, 20,140 images = 4,028 references + 16,112 twins) and `vp_pairs_v1_baselines.zip` (443 MB, 4,768 images), plus checks14.json, jpeg_rescore.json (JPEG re-scoring shifts G: median |dG| 0.018, max 0.198).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `build.py`, `check.py`, `readme.py`.

# task06: Fixed preprocessing before DeepLSD + new rotation rule (2026-11-15)

`rot6.py` defines the rotation rule used since (2-deg drop significant and 4/8-deg drops not significantly smaller; now `eval/metrics.rotation_test`). `pipe6.py` ran DeepLSD on 57,980 images for 3 pre-blur levels (2.6 GPU-h) and scored them; `eval6.py` evaluates. Several settings passed on SELECT; the pick failed the rotation rule on Hypersim REPORT. Outputs `$VP_DATA/outputs/task6/` (eval6_*, pass6_*, exploit6.csv, scores6_raw.pkl).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `eval6.py`, `pipe6.py`, `rot6.py`.

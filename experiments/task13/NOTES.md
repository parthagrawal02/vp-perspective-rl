# task13: Midway-report figures (2026-11-20)

`figlib.py` (overlay helpers; re-runs the frozen fit and asserts it equals `final.geometry` bit-for-bit) and `make_figs.py` built figures A1, A2, B, C, D, E1, E2, F, G from existing images and cached DeepLSD lines only (CPU, ~1 min). Selection seeds 1301-1306 are in the script and the captions. Outputs `$VP_DATA/outputs/report_figs/` (*.jpg + caption .txt, README.md, e2_long_short.json, e2_pairs.csv).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `figlib.py`, `make_figs.py`.

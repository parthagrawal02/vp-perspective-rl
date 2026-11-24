# task09: Guards + combined-reward exploit test (2026-11-16)

`data9.py` computed PickScore + aesthetic (GPU) and G/Lc; `eval9.py` grid-searched R = zG - lf*P + lq*zQ (floor alpha, lf, quality term) against the exploit test, rotation, learnability (`learn`) and the within-caption Spearman with G (`geo_corr`); both now in `eval/metrics.py`. Pick: p2 sig, floor alpha 0.9, lf 1, Q = sum, lq 0.25. Outputs `$VP_DATA/outputs/task9/` (grid9.csv, pick_decomp.csv, table9_data.pkl).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `data9.py`, `eval9.py`.

# task10: Contrast stretch, no blur; standardisation constants (2026-11-17)

`v4.stretch` (1-99 percentile stretch, the frozen preprocessing). `pipe10.py` ran DeepLSD on 32,000 SELECT-side + exploit images (GPUs 1,3,5,7) and scored them; `eval10.py` sanity + a 92-setting grid. The z-score constants of FINAL.md section 4 come from these SELECT base twins (n = 6,800). Stopped per rule (0 acceptable settings at the 20% floor-activity cap); the chosen setting was frozen in Task 11 with the cap relaxed to 23% (recorded in FINAL.md). Outputs `$VP_DATA/outputs/task10/` (g10.pkl, table10_data.pkl, grid10.csv, sanity10.csv, exploitG10.csv, dlsd10_*.pkl).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `eval10.py`, `pipe10.py`.

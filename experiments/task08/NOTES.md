# task08: Continuous pooled score (2026-11-15)

`v4.compute_t8` (the continuous p2-sig score later frozen as G), `pipe8.py` (CPU scoring on cached Task 3/6 lines), `eval8.py` (adds the untied win rate and 2-deg sensitivity shares; now in `eval/metrics.py`). No combination passed the exploit test alone. Outputs `$VP_DATA/outputs/task8/` (table8.csv, eval8_SELECT.csv, exploit8.csv).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `eval8.py`, `pipe8.py`.

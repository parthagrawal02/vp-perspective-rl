# task07: Ties + sensitivity diagnostics (2026-11-15)

`diag7.py` re-computed real and rotated images (CPU) to count ties and per-image sensitivity; recomputed scores identical to saved. Ties were already counted as 1/2 in all win rates since Task 3. Outputs `$VP_DATA/outputs/task7/` (ties.csv, sens.csv).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `diag7.py`.

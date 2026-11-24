# task05: Top-k segments per VP cluster (2026-11-15)

`v4.compute_t5` / `topk_scores`; `score5.py` (CPU), `eval5.py`. No variant passed: all 8 fail the exploit test (blur, noise 10) and the Hypersim rotation test as defined then. Outputs `$VP_DATA/outputs/task5/` (eval5_SELECT.csv, pass5_SELECT.csv, exploit.csv, decomp.csv).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `eval5.py`, `score5.py`.

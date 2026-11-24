# task04: Blur-proof A1 variants + exploit test (2026-11-15)

`vp_reward/v4.py` variants V0-V3 (V2 = fixed sigma0, length-only uncertainty). `score4.py` re-scored all Task 3 data and ran DeepLSD on perturbed twins (blur, contrast, rescale, noise, JPEG); `eval4.py` adds the degraded-twin test and the exploit test (a perturbation that raises the score significantly is an exploit). Result: no variant works; all 10 fail the exploit test because DeepLSD drops weak/texture segments on degraded images and the survivors fit better (detection-side exploit, `diag4.py`). Outputs `$VP_DATA/outputs/task4/` (eval4_SELECT.csv, exploit.csv, diag_decompose.csv, exdlsd_*.pkl).

Scripts as run, with light fixes only: hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir(...)` for sibling code) / `config/env.sh`, and `vp_reward` imported as the installed package. Files: `diag4.py`, `eval4.py`, `score4.py`.

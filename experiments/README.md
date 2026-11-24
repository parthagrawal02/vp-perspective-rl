# experiments/: archive of every task as run

One folder per task (task01 … task14). Each holds that task's scripts **exactly as they were run**, with light fixes only:
hardcoded paths replaced by `vp_reward.paths` (`DATA` = `$VP_DATA`, `code_dir("taskN")` for sibling task code) or
`config/env.sh` in shell scripts, and `vp_reward` imported as the installed package. Each folder's `NOTES.md` says what the task
did, its key result and where its outputs went (all under `$VP_DATA/outputs/`). Scripts import siblings from other task folders
(e.g. Task 11 evaluation reuses Task 3/4/6/8/9 code), so keep the layout. They were run with the `nft`, `nft-vp`, `deeplsd` or
`vlm` env (see each script's docstring / NOTES). The shared, cleaned versions of the reused pieces are in `vp_reward/`, `eval/`,
`data/` and `rl/`.

| task | topic | | task | topic |
|---|---|---|---|---|
| 01 | env + DrawBench baseline replication | | 08 | continuous pooled score (→ G) |
| 02 | first VP reward, NYU v2 validation | | 09 | guards: floor + quality terms |
| 03 | DeepLSD, SUN RGB-D + Hypersim, variants | | 10 | contrast stretch, standardisation constants |
| 04 | blur-proof variants, exploit test | | 11 | **freeze**, fresh Hypersim split, baselines |
| 05 | top-k segments per VP | | 12 | DiffusionNFT wiring + smoke test |
| 06 | preprocessing, rotation rule | | 13 | midway-report figures |
| 07 | ties + sensitivity diagnostics | | 14 | dataset package vp_pairs_v1 |

# eval/

`metrics.py`: the validation criteria used to select and check the reward (Tasks 3–11), as plain functions over a long table
(one row per image × RANSAC seed; columns documented at the top of the file). Logic copied unchanged from the task scripts; on the
Task 11 table all 18 compared values (win rates, quintiles, rotation p-values, learnability, degraded-twin test, 2-deg shares)
are bit-identical to the original code.

| function | criterion (pass rule used in the project) |
|---|---|
| `paired_win_rate`, `untied_win_rate` | real reference beats its caption-matched twins (ties ½); > 0.5 with Wilcoxon p < 0.01 |
| `lc_quintile_win_rates` | same, within quintiles of structure amount Lc; ≥ 4 of 5 quintiles > 0.5 |
| `rotation_test` | 2° rotation of a region lowers the score (p < 0.01), 4° and 8° not significantly less than 2° |
| `degraded_twin_test` | reference beats twins degraded to its sharpness; > 0.5, p < 0.01 |
| `exploit_test` | no perturbation (blur, noise, contrast, rescale, JPEG) raises the score significantly (one-sided p < 0.01) |
| `learnability` | caption-level spread / RANSAC-seed noise ≥ 3 |
| `within_caption_spearman`, `floor_activity` | R ranks like G within a caption; share of base twins with P > 0 |

`python -m eval.fresh_split` (repo root, CPU, < 1 min) recomputes the Task 11 headline from
`$VP_DATA/outputs/task11/table11_data.pkl`: G win rate 0.688, quintiles 0.57 / 0.74 / 0.70 / 0.82 / 0.73, rotation pass,
learnability 6.06, degraded-twin 0.674, and the R exploit table (noise 2 +0.043* fails).

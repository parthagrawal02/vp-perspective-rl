"""Reproduce the fresh-Hypersim-split validation headline (Task 11, FINAL.md §7) from the stored per-image scores.
    python -m eval.fresh_split            (from the repo root; CPU, ~1 min)
Input  : $VP_DATA/outputs/task11/table11_data.pkl  (one row per image x RANSAC seed: G, Lc, PickScore, aesthetic, Lc_base, R, ...;
         built by experiments/task11/eval11.py from g11_*.pkl / q11_*.pkl)
         $VP_DATA/outputs/task11/winlen11.pkl     (rotation crop window per reference)
Output : printed table; compares against the stored $VP_DATA/outputs/task11/confirm11.csv."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

from vp_reward.paths import DATA
from . import metrics as M

O = os.path.join(DATA, "outputs", "task11")
EXPECTED = {"wr_untied": 0.688, "quintiles": [0.57, 0.74, 0.70, 0.82, 0.73]}   # FINAL.md §7 / confirm11.csv (rounded)


def main():
    T = pd.read_pickle(f"{O}/table11_data.pkl"); wl = pickle.load(open(f"{O}/winlen11.pkl", "rb"))
    Mm = T[T.kind == "m"]                                      # main rows: real, base twins, degraded twins, rotations
    wr, ties = M.untied_win_rate(Mm, "G"); pw = M.paired_win_rate(Mm, "G"); q = M.lc_quintile_win_rates(Mm, "G")
    rot = M.rotation_test(Mm, "G", wl); deg = M.degraded_twin_test(Mm, "G"); lrn = M.learnability(Mm, "G")
    X = T[(T.kind == "x") & (T.rseed == 0)].assign(ds="hf"); ex = M.exploit_test(X, "R")
    rows = [("G win rate (untied; ties)", f"{wr:.3f} ({ties:.3f}); paired wr {pw['wr']:.3f}, p = {pw['p']:.2g}, n = {pw['n_pairs']}"),
            ("G Lc-quintile win rates", ", ".join(f"{q[f'wr_q{k}']:.2f}" for k in range(1, 6)) + f"  ({q['q_above']}/5 > 0.5)"),
            ("G rotation rule", f"pass {rot['rot_pass']}: p_rot2 {rot['p_rot2']:.2g}, p(4<2) {rot['p_4lt2']:.2g}, p(8<2) {rot['p_8lt2']:.2g}"),
            ("G learnability", f"{lrn:.2f}"),
            ("G degraded-twin win rate", f"{deg['wr_deg']:.3f}, p = {deg['p_deg']:.2g}"),
            ("R exploit (mean dR, * = sig. up)", ", ".join(f"{k} {v[0]:+.3f}{'*' if v[2] else ''}" for k, v in ex.items())),
            ("R floor activity (base twins)", f"{M.floor_activity(Mm):.1%}")]
    for k, v in rows: print(f"{k:34s} {v}")
    got_q = [round(q[f"wr_q{k}"], 2) for k in range(1, 6)]
    ok = round(wr, 3) == EXPECTED["wr_untied"] and got_q == EXPECTED["quintiles"]
    if os.path.exists(f"{O}/confirm11.csv"):
        C = pd.read_csv(f"{O}/confirm11.csv"); print("\nstored confirm11.csv:"); print(C.head(5).to_string(index=False))
    print(f"\nheadline check: G win rate {wr:.3f} (expected {EXPECTED['wr_untied']}), quintiles {got_q} (expected {EXPECTED['quintiles']}) ->",
          "MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

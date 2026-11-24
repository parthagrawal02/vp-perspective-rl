"""Reproduction check (Task 11 / Task 12): score 20 saved SELECT base twins (+1 twin with undefined G) with the full reward and
compare with the values stored by Task 10. Needs the stored outputs under $VP_DATA and a GPU; skipped otherwise.
Expected: G and Lc identical (0 difference); PickScore / aesthetic / R within float32 scorer noise (Task 11: R 1.2e-6)."""
import os
import pickle

import numpy as np
import pytest

from vp_reward.paths import DATA

SAMPLE = os.path.join(DATA, "outputs", "task11", "repro_sample.pkl")
torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.data, pytest.mark.gpu,
              pytest.mark.skipif(not os.path.exists(SAMPLE), reason=f"no stored sample at {SAMPLE}"),
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")]


def _local(p):  # stored paths are absolute on the machine that made them; re-root them under $VP_DATA
    return os.path.join(DATA, p[p.index("outputs/"):])


def test_reproduce_task10_values():
    from vp_reward.score import Scorer
    S = pickle.load(open(SAMPLE, "rb"))
    out = Scorer("cuda", n_workers=0)([_local(p) for p in S["path"]], list(S["prompt"]), np.array(S["Lc_base"], float))
    ref = {k: np.array(S[k], float) for k in ("G_p2", "Lc", "Q_pick", "Q_aes", "R_t10")}
    d = {"G": np.abs(out["G"][:20] - ref["G_p2"][:20]).max(), "Lc": np.abs(out["Lc"][:20] - ref["Lc"][:20]).max(),
         "PickScore": np.abs(out["pickscore"][:20] - ref["Q_pick"][:20]).max(), "aesthetic": np.abs(out["aesthetic"][:20] - ref["Q_aes"][:20]).max(),
         "R": np.abs(out["R"][:20] - ref["R_t10"][:20]).max()}
    print("\nmax |diff| over 20 images:", {k: float(f"{v:.3g}") for k, v in d.items()})
    print("NaN-G twin: G_nan", bool(out["G_nan"][20]), "G", out["G"][20], "(stored G", S["G_p2"][20], ")")
    assert d["G"] == 0 and d["Lc"] == 0
    assert d["PickScore"] < 1e-4 and d["aesthetic"] < 1e-5 and d["R"] < 1e-5
    assert bool(out["G_nan"][20]) and out["G"][20] == 0.37950405006078053

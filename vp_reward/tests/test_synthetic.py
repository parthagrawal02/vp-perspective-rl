import numpy as np
import pytest

from vp_reward import VPConfig, detect_segments, fit_and_score
from vp_reward.synth import bend, render, scene

CFG = VPConfig()


def _score(groups, seed=0):
    img = render(groups, seed)
    return fit_and_score(detect_segments(img), CFG, seed)


@pytest.mark.parametrize("kind", ["one", "two"])
def test_clean_scene_near_perfect(kind):
    r = _score(scene(kind, seed=1))
    s = r.scores
    print(kind, s, "nvps", len(r.diag["vps"]), "R_mean", r.diag["R_mean"])
    assert len(r.diag["vps"]) == 3
    assert s["G"] > 0.9
    assert s["NM"] > 0.95
    assert s["RegMean"] > 0.75  # R_mean < ~0.6 deg
    assert s["RegMax"] > 0.4  # worst tile; the tile containing a finite VP is ill-conditioned (~1.3 deg)


@pytest.mark.parametrize("kind", ["one", "two"])
def test_bend_monotone(kind):
    base = scene(kind, seed=2)
    deltas = [0, 1, 2, 4, 8]
    res = {k: [] for k in ("G", "NM", "RegMean", "RegMax")}
    for d in deltas:
        g, sel = bend(base, d)
        sc = _score(g).scores
        for k in res:
            res[k].append(sc[k])
    print(kind, {k: np.round(v, 3).tolist() for k, v in res.items()})
    for k in ("G", "RegMean", "RegMax"):
        v = np.array(res[k])
        assert v[-1] < v[0] - 0.02, k
        # roughly monotone: allow small non-monotonic noise
        assert np.all(np.diff(v) < 0.03), (k, v)

"""Unit tests of the EXPERIMENTAL v2 reward (vp_reward/reward_v2.py). CPU only (ground-truth synthetic segments, no DeepLSD)."""
import numpy as np
import pytest

from vp_reward import final as F
from vp_reward import reward_v2 as V
from vp_reward.synth import bend, render, scene


def _pre_lines(groups, seed=0):
    return F.stretch(render(groups, seed)), np.concatenate(groups).astype(np.float32)


@pytest.mark.parametrize("kind,d", [("one", 0), ("two", 4), ("one", 8)])
def test_beta0_equals_frozen_geometry(kind, d):
    pre, lines = _pre_lines(bend(scene(kind, seed=2), d)[0])
    assert V.geometry(pre, lines, beta=0.0) == F.geometry(pre, lines)


def test_beta_only_lowers_G_and_keeps_Lc():
    rng = np.random.default_rng(0)
    groups = scene("one", seed=3) + [rng.uniform(40, 470, (25, 4))]   # random clutter -> some unassigned segments
    f = V.fit(*_pre_lines(groups))
    assert (~f["asg"]).any()
    G0, L0 = V.score(f, 0.0); G1, L1 = V.score(f, 0.5); G2, L2 = V.score(f, 1.0)
    assert G2 < G1 < G0 and L0 == L1 == L2


def test_cap_P():
    Ph = V.cap_P([1000.0, 1750.0, 2625.0, 5000.0, 10.0], [1000.0] * 4 + [0.0], gamma=1.75)
    np.testing.assert_allclose(Ph, [0.0, 0.0, 0.5, 1.0, 0.0])


def test_combine_matches_frozen_when_cap_inactive():
    G = np.array([0.8, np.nan, 0.5]); Lc = np.array([1000.0, 0.0, 300.0]); Lb = np.array([1000.0, 900.0, 1000.0])
    pick = np.array([23.0, 22.0, 21.0]); aes = np.array([5.5, 5.0, 6.0])
    R1, _ = F.combine(G, Lc, Lb, pick, aes)
    R2, i2 = V.combine(G, Lc, Lb, pick, aes, beta=0.0)
    np.testing.assert_allclose(R1, R2, atol=1e-12)
    assert (i2["P_hi"] == 0).all()


def test_combine_cap_penalty():
    R1, _ = F.combine([0.8], [3500.0], [1000.0], [F.PICK_MEAN], [F.AES_MEAN])
    R2, i2 = V.combine([0.8], [3500.0], [1000.0], [F.PICK_MEAN], [F.AES_MEAN], beta=0.0, gamma=1.75, lambda_hi=4.0)
    assert i2["P_hi"][0] == 1.0 and np.isclose(R2[0], R1[0] - 4.0)


def test_unknown_beta_needs_constants():
    with pytest.raises(KeyError):
        V.combine([0.8], [1.0], [1.0], [F.PICK_MEAN], [F.AES_MEAN], beta=0.123)


def test_default_is_frozen_G_plus_cap():
    assert V.BETA == 0.0 and V.G_STATS[0.0] == (F.G_MEAN, F.G_STD, F.G_NAN_FILL)

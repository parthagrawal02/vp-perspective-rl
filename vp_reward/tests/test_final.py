"""Unit tests of the frozen reward (vp_reward/final.py, FINAL.md). CPU only: geometry is fed the ground-truth segments of a
synthetic scene instead of DeepLSD lines, so no network or weights are needed."""
import numpy as np
import pytest

from vp_reward import final as F
from vp_reward.synth import bend, render, scene


def test_frozen_constants_match_final_md():
    assert (F.G_MEAN, F.G_STD) == (0.6762756750467267, 0.12087393883907192)
    assert (F.PICK_MEAN, F.PICK_STD) == (22.55044470282162, 0.989595540429443)
    assert (F.AES_MEAN, F.AES_STD) == (5.452365678899429, 0.3416999366620726)
    assert (F.Q_SUM_MEAN, F.Q_SUM_STD) == (5.684341886080802e-16, 1.5849401901583253)
    assert F.G_NAN_FILL == 0.37950405006078053
    assert (F.ALPHA, F.LAMBDA_F, F.LAMBDA_Q) == (0.8, 4.0, 0.25)
    assert (F.LMIN_FRAC, F.SIGMA0, F.RANSAC_SEED) == (0.04, 1.0, 0)
    assert F.DEEPLSD_CONF == {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}


def test_combine_formula_and_nan_rule():
    G = np.array([0.8, np.nan, 0.5]); Lc = np.array([1000.0, 0.0, 300.0]); Lb = np.array([1000.0, 900.0, 1000.0])
    pick = np.array([23.0, 22.0, 21.0]); aes = np.array([5.5, 5.0, 6.0])
    R, info = F.combine(G, Lc, Lb, pick, aes)
    assert info["G_nan"].tolist() == [False, True, False] and info["G"][1] == F.G_NAN_FILL
    zG = (info["G"] - F.G_MEAN) / F.G_STD
    P = np.maximum(0, 1 - Lc / (0.8 * Lb))
    zQ = ((pick - F.PICK_MEAN) / F.PICK_STD + (aes - F.AES_MEAN) / F.AES_STD - F.Q_SUM_MEAN) / F.Q_SUM_STD
    np.testing.assert_allclose(R, zG - 4 * P + 0.25 * zQ, rtol=0, atol=1e-12)
    assert P[0] == 0 and P[1] == 1 and np.isclose(P[2], 1 - 300 / 800)


def test_floor_zero_when_no_reference():
    assert F.floor_P([10.0], [0.0])[0] == 0.0 and F.floor_P([10.0], [-1.0])[0] == 0.0


def test_stretch_maps_percentiles():
    rng = np.random.default_rng(0)
    img = rng.integers(60, 120, (512, 512, 3)).astype(np.uint8)
    s = F.stretch(img)
    assert s.dtype == np.float32 and s.min() == 0 and s.max() == 255
    assert np.isclose(np.percentile(s, 1), 0, atol=3) and np.isclose(np.percentile(s, 99), 255, atol=3)


def _geo(groups, seed=0):
    img = render(groups, seed)
    return F.geometry(F.stretch(img), np.concatenate(groups).astype(np.float32))


@pytest.mark.parametrize("kind", ["one", "two"])
def test_geometry_clean_scene(kind):
    G, Lc = _geo(scene(kind, seed=1))
    assert G > 0.9 and Lc > 5000


@pytest.mark.parametrize("kind", ["one", "two"])
def test_geometry_bend_lowers_G(kind):
    base = scene(kind, seed=2)
    Gs = [_geo(bend(base, d)[0])[0] for d in (0, 4, 8)]
    assert Gs[2] < Gs[0] - 0.02 and Gs[1] < Gs[0], Gs   # same margin as test_synthetic.py


def test_geometry_deterministic_and_nan():
    g = scene("one", seed=3)
    assert _geo(g) == _geo(g)
    G, Lc = F.geometry(F.stretch(render(g)), np.zeros((0, 4), np.float32))   # no lines -> undefined G, Lc 0
    assert np.isnan(G) and Lc == 0

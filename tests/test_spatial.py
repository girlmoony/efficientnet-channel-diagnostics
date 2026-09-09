import numpy as np
import pytest
import torch
from efficientnet_diagnostics import diagnose, validate_regions
from efficientnet_diagnostics.spatial import concentration, hotspot, nearest
from test_diagnostics import Toy, sample


def test_concentration_edge_cases():
    assert concentration(np.zeros((8,8)))["entropy"] is None
    uniform = concentration(np.ones((8,8)))
    assert uniform["entropy"] == pytest.approx(1)
    assert uniform["max_share"] == 1/64
    assert uniform["top4_share"] == 4/64
    spike = np.zeros((8,8)); spike[2,3] = 5
    metrics = concentration(spike)
    assert metrics == dict(max_share=1., top4_share=1., entropy=0., positive_area=1/64)
    assert hotspot(np.ones((8,8))).sum() == 64  # Do not hide ties.
    assert hotspot(-np.ones((8,8))).sum() == 0
    assert nearest(spike, (256,256))[64:96,96:128].min() == 5


def test_fixed_pair_equal_area_intervention():
    model = Toy().eval()
    x = sample()
    r = diagnose(model, x, 0)
    data, artifacts = validate_regions(model, x, r, top_k=1)
    assert (data["A"],data["B"]) == (0,1)
    hot, _ = artifacts["channel_0_mean_hot"]
    low, _ = artifacts["channel_0_mean_low"]
    assert hot.sum() == low.sum() == 8
    assert not (hot & low).any()
    row = next(row for row in data["rows"] if row["status"] == "comparison" and row["target"] == "channel_0" and row["method"] == "mean")
    assert row["hot_minus_low_drop"] > 0
    assert all(row["logit_B"] - row["logit_A"] == pytest.approx(row["margin"]) for row in data["rows"] if row["status"] == "ok")


def test_uniform_hotspot_skips_control():
    model = Toy().eval()
    x = torch.zeros(1,3,4,4); x[:,0] = 1
    r = diagnose(model,x,0)
    data, artifacts = validate_regions(model,x,r)
    assert not artifacts
    assert all(row["status"] == "skipped" for row in data["rows"])

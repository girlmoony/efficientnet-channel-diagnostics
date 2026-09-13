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


@pytest.mark.parametrize("true_class,channel", [(0, 0), (1, 1)])
def test_fixed_pair_equal_area_intervention(true_class, channel):
    model = Toy().eval()
    with torch.no_grad():
        model.classifier.bias[1] = -.6  # Replacements can change Top-1.
    x = sample()
    r = diagnose(model, x, true_class)
    data, artifacts = validate_regions(model, x, r, top_k=1)
    assert (data["A"],data["B"]) == (true_class, 1-true_class)
    assert data["predicted_class"] == 1
    assert data["comparison_class"] == 1-true_class
    target = f"channel_{channel}"
    hot, _ = artifacts[f"{target}_mean_hot"]
    low, _ = artifacts[f"{target}_mean_low"]
    assert hot.sum() == low.sum() == 8
    assert not (hot & low).any()
    row = next(row for row in data["rows"] if row["status"] == "comparison" and row["target"] == target and row["method"] == "mean")
    assert row["hot_minus_low_drop"] > 0
    assert all(row["logit_B"] - row["logit_A"] == pytest.approx(row["margin"]) for row in data["rows"] if row["status"] == "ok")
    assert any(row["predicted_class"] != r.predicted_class
               for row in data["rows"] if row["status"] == "ok")
    for row in data["rows"]:
        if row["status"] != "ok":
            continue
        _, modified = artifacts[row["artifact"]]
        with torch.no_grad():
            logits = model(modified[None])[0]
        assert row["predicted_class"] == int(logits.argmax())
        assert row["comparison_class"] == r.comparison_class
        assert row["logit_B"] == float(logits[r.comparison_class])
        assert row["logit_A"] == float(logits[r.true_class])
        assert row["margin_drop"] == pytest.approx(data["original_margin"] - row["margin"])


def test_uniform_hotspot_skips_control():
    model = Toy().eval()
    x = torch.zeros(1,3,4,4); x[:,0] = 1
    r = diagnose(model,x,0)
    data, artifacts = validate_regions(model,x,r)
    assert not artifacts
    assert all(row["status"] == "skipped" for row in data["rows"])

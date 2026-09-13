import json
import re
import sys

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from efficientnet_diagnostics import diagnose, save_report, validate_regions
from efficientnet_diagnostics import cli
from efficientnet_diagnostics.cli import load_weights, preprocess


class Toy(nn.Module):
    """Known spatial evidence: red input channel pushes B; green pushes A."""
    def __init__(self, bias=True):
        super().__init__()
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.classifier = nn.Linear(3, 2, bias=bias)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[0., 1., 0.], [2., 0., 0.]]))
            if bias:
                self.classifier.bias.copy_(torch.tensor([0., .25]))

    def forward(self, x):
        return self.classifier(self.global_pool(x))


def sample():
    x = torch.zeros(1, 3, 4, 4)
    x[:, 0, :, 2:] = 1  # Right side pushes B, contribution +1
    x[:, 1, :, :2] = .5  # Left side pushes A, contribution -.25
    return x


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("true_class", [0, 1])
def test_known_contributions_regions_and_ablation(bias, true_class):
    model = Toy(bias).eval()
    x = sample()
    r = diagnose(model, x, true_class)
    sign = 1 if true_class == 0 else -1
    torch.testing.assert_close(r.margin_contribution, sign * torch.tensor([1., -.25, 0.]))
    assert r.predicted_class == 1
    assert r.comparison_class == 1 - true_class
    assert r.share_b.tolist() == ([1., 0., 0.] if sign == 1 else [0., 1., 0.])
    assert r.share_a.tolist() == ([0., 1., 0.] if sign == 1 else [1., 0., 0.])
    assert torch.all(r.channel_maps.sum(0)[:, 2:] == sign * 2)
    assert torch.all(r.channel_maps.sum(0)[:, :2] == sign * -.5)
    torch.testing.assert_close(r.channel_maps.mean((1, 2)), r.margin_contribution)
    torch.testing.assert_close(r.margin_contribution.sum() + r.bias_margin,
                               r.logits[r.comparison_class] - r.logits[r.true_class])
    h = model.global_pool(x)
    before = model.classifier(h)[0].diff()[0]
    h[:, 0] = 0
    after = model.classifier(h)[0].diff()[0]
    torch.testing.assert_close(sign * (before - after), r.margin_contribution[0])


def test_reject_invalid_modes_and_pool():
    model = Toy()
    with pytest.raises(ValueError, match="eval"):
        diagnose(model, sample(), 0)
    model.eval()
    with pytest.raises(ValueError, match="index"):
        diagnose(model, sample(), -1)
    model.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(1))
    model.eval()
    with pytest.raises(AssertionError):
        diagnose(model, sample(), 0)
    assert not model.classifier._forward_pre_hooks
    assert not model.global_pool._forward_pre_hooks


@pytest.mark.parametrize("true_class", [0, 1])
@pytest.mark.parametrize("with_intervention", [False, True])
def test_report_and_alignment(tmp_path, true_class, with_intervention):
    model = Toy().eval()
    r = diagnose(model, sample(), true_class)
    rgb = sample()[0].permute(1, 2, 0).numpy()
    intervention = validate_regions(model, sample(), r) if with_intervention else None
    report = save_report(r, rgb, tmp_path / "report", class_names=["<A>", "B"],
                         intervention=intervention, normalization=([0]*3, [1]*3))
    page = report.read_text(encoding="utf-8")
    assert "&lt;A&gt;" in page
    assert '<html lang="en">' in page
    assert "predicted Top-1" in page and "comparison class B" in page
    assert ("runner-up" if true_class == 1 else "incorrect Top-1") in page
    assert not re.search(r"[\u3400-\u9fff]", page)
    assert len((report.parent / "channels.csv").read_text().splitlines()) == 4
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["predicted_class"] == 1
    assert summary["comparison_class"] == 1 - true_class
    assert summary["true_class"] == true_class
    assert summary["logit_B"] == float(r.logits[r.comparison_class])
    assert summary["margin_B_minus_A"] == (1 if true_class == 0 else -1)
    for name in ["channel_distribution.png", "total_regions.png", "channels_push_A.png",
                 "channels_push_B.png", "channel_0_details.png", "channel_1_details.png",
                 "spatial_metrics.csv", "spatial_metrics.json", "input.png"]:
        assert (report.parent / name).is_file()
    metrics = json.loads((report.parent / "spatial_metrics.json").read_text())
    assert len(metrics) == 3
    assert metrics[0]["B_hot_grid_fraction" if true_class == 0 else "A_hot_grid_fraction"] == .5
    if with_intervention:
        data = json.loads((report.parent / "interventions.json").read_text())
        assert data["predicted_class"] == 1
        assert data["comparison_class"] == 1 - true_class
        assert data["original_margin"] == summary["margin_B_minus_A"]
        assert "Hot versus low-contribution region controls" in page
        successful = [row for row in data["rows"] if row["status"] == "ok"]
        assert successful
        for row in successful:
            assert row["comparison_class"] == r.comparison_class
            assert (report.parent / (row["artifact"] + ".png")).is_file()
            assert (report.parent / (row["artifact"] + "_mask.png")).is_file()
    maps = np.load(report.parent / "spatial_contributions.npz")
    np.testing.assert_allclose(maps["total_map"], r.channel_maps.sum(0).numpy())
    np.testing.assert_allclose(maps["feature_maps"], r.feature_maps.numpy())
    for path in report.parent.glob("*.png"):
        with Image.open(path) as image:
            image.verify()
    with pytest.raises(ValueError, match="empty"):
        save_report(r, rgb, report.parent)
    with pytest.raises(ValueError, match="match actual"):
        save_report(r, np.zeros((8, 8, 3)), tmp_path / "bad")


def test_checkpoint_and_preprocessing(tmp_path):
    model = Toy().eval()
    checkpoint = tmp_path / "weights.pth"
    torch.save({"state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, checkpoint)
    restored = Toy().eval()
    load_weights(restored, checkpoint)
    torch.testing.assert_close(restored(sample()), model(sample()))
    image = tmp_path / "image.png"
    Image.fromarray(np.full((4, 4, 3), 255, np.uint8)).save(image)
    rgb, x = preprocess(image, 4, [.5]*3, [.5]*3, "none", "bilinear")
    assert np.all(rgb == 1) and torch.all(x == 1)
    with pytest.raises(ValueError, match="size"):
        preprocess(image, 8, [.5]*3, [.5]*3, "none", "bilinear")


@pytest.mark.parametrize("correct", [False, True])
def test_real_timm_lite1_structure(correct):
    import timm
    torch.manual_seed(17)
    model = timm.create_model("tf_efficientnet_lite1", pretrained=False, num_classes=400).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        predicted = int(model(x).argmax())
    r = diagnose(model, x, predicted if correct else (predicted + 1) % 400)
    assert r.predicted_class == predicted
    alternatives = r.logits.clone()
    alternatives[r.true_class] = -torch.inf
    assert r.comparison_class == int(alternatives.argmax())
    assert r.channel_maps.shape == (1280, 8, 8)
    assert r.margin_contribution.shape == (1280,)
    assert r.reconstruction_error < 1e-4


@pytest.mark.parametrize("logits,true_class,predicted,comparison", [
    ([-3., -1., -2.], 1, 1, 2),  # Runner-up is selected by logit, not index.
    ([3., 1., 2.], 1, 0, 0),    # Incorrect Top-1 stays B.
    ([2., 3., 2.], 1, 1, 0),    # Tied runner-ups choose the first index.
    ([3., 3., 1.], 0, 0, 1),    # Correct Top-1 tie has zero margin.
    ([3., 3., 1.], 1, 0, 0),    # Incorrect Top-1 tie also has zero margin.
])
def test_multiclass_selection(logits, true_class, predicted, comparison):
    model = Toy().eval()
    model.classifier = nn.Linear(3, 3).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor(logits))
    r = diagnose(model, sample(), true_class)
    assert (r.predicted_class, r.comparison_class) == (predicted, comparison)
    assert r.comparison_class != r.true_class
    assert r.bias_margin == logits[comparison] - logits[true_class]
    assert r.reconstruction_error == 0


def test_reject_single_class():
    model = Toy().eval()
    model.classifier = nn.Linear(3, 1).eval()
    with pytest.raises(ValueError, match="at least two classes"):
        diagnose(model, sample(), 0)


@pytest.mark.parametrize("true_class", [0, 1])
def test_cli_reports_prediction_and_comparison(tmp_path, monkeypatch, capsys, true_class):
    checkpoint = tmp_path / "weights.pth"
    torch.save(Toy().state_dict(), checkpoint)
    image = tmp_path / "input.png"
    Image.fromarray((sample()[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(image)
    output = tmp_path / "report"
    monkeypatch.setattr(cli.timm, "create_model", lambda *args, **kwargs: Toy())
    monkeypatch.setattr(sys, "argv", ["efficientnet-diagnose", "--checkpoint", str(checkpoint),
        "--image", str(image), "--true-class", str(true_class), "--num-classes", "2",
        "--size", "4", "--mean", "0", "0", "0", "--std", "1", "1", "1",
        "--output", str(output), "--validate-regions", "--validation-top-k", "1"])
    cli.main()
    stdout = capsys.readouterr().out
    assert f"True class A={true_class}, predicted_class=1, comparison_class B={1-true_class}" in stdout
    assert "Report:" in stdout
    data = json.loads((output / "summary.json").read_text())
    assert data["predicted_class"] == 1 and data["comparison_class"] == 1-true_class
    assert (output / "interventions.json").is_file()

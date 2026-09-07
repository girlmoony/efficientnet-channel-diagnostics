import json

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from efficientnet_diagnostics import diagnose, save_report
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
def test_known_contributions_regions_and_ablation(bias):
    model = Toy(bias).eval()
    x = sample()
    r = diagnose(model, x, 0)
    torch.testing.assert_close(r.margin_contribution, torch.tensor([1., -.25, 0.]))
    assert r.predicted_class == 1
    assert r.share_b.tolist() == [1., 0., 0.]
    assert r.share_a.tolist() == [0., 1., 0.]
    assert torch.all(r.channel_maps.sum(0)[:, 2:] == 2)
    assert torch.all(r.channel_maps.sum(0)[:, :2] == -.5)
    h = model.global_pool(x)
    before = model.classifier(h)[0].diff()[0]
    h[:, 0] = 0
    after = model.classifier(h)[0].diff()[0]
    torch.testing.assert_close(before - after, r.margin_contribution[0])


def test_reject_invalid_modes_and_pool():
    model = Toy()
    with pytest.raises(ValueError, match="eval"):
        diagnose(model, sample(), 0)
    model.eval()
    with pytest.raises(ValueError, match="no A-to-B"):
        diagnose(model, sample(), 1)
    with pytest.raises(ValueError, match="index"):
        diagnose(model, sample(), -1)
    model.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(1))
    model.eval()
    with pytest.raises(AssertionError):
        diagnose(model, sample(), 0)
    assert not model.classifier._forward_pre_hooks
    assert not model.global_pool._forward_pre_hooks


def test_report_and_alignment(tmp_path):
    r = diagnose(Toy().eval(), sample(), 0)
    rgb = sample()[0].permute(1, 2, 0).numpy()
    report = save_report(r, rgb, tmp_path / "report", class_names=["<A>", "B"])
    assert "&lt;A&gt;" in report.read_text(encoding="utf-8")
    assert len((report.parent / "channels.csv").read_text().splitlines()) == 4
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["margin_B_minus_A"] == 1
    maps = np.load(report.parent / "spatial_contributions.npz")
    np.testing.assert_allclose(maps["total_map"], r.channel_maps.sum(0).numpy())
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


def test_real_timm_lite1_structure():
    import timm
    torch.manual_seed(17)
    model = timm.create_model("tf_efficientnet_lite1", pretrained=False, num_classes=400).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        predicted = int(model(x).argmax())
    r = diagnose(model, x, (predicted + 1) % 400)
    assert r.channel_maps.shape == (1280, 8, 8)
    assert r.margin_contribution.shape == (1280,)
    assert r.reconstruction_error < 1e-4

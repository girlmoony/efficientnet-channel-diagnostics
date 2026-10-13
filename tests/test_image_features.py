"""Regression coverage for image-version features and RGB/BGR rendering."""
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from efficientnet_diagnostics import diagnose, save_report, validate_regions
from efficientnet_diagnostics import cli
from efficientnet_diagnostics.cli import load_labels, preprocess
from test_diagnostics import Toy, sample


@pytest.mark.parametrize('true_class', [0, 1])
def test_image_api_names_preserve_existing_results(true_class):
    result = diagnose(Toy().eval(), sample(), true_class)
    assert result.contrast_class == result.comparison_class == 1 - true_class
    assert result.is_correct is (true_class == 1)
    torch.testing.assert_close(result.margin_contribution.sum() + result.bias_margin,
                               result.logits[result.contrast_class] - result.logits[true_class])


@pytest.mark.parametrize('order', ['rgb', 'bgr'])
def test_preprocess_channel_order_and_normalization(tmp_path, order):
    rgb = np.full((4, 4, 3), [51, 102, 204], dtype=np.uint8)
    path = tmp_path / 'input.png'
    Image.fromarray(rgb).save(path)
    mean, std = [.1, .2, .3], [.2, .4, .5]
    shown, tensor = preprocess(path, 4, mean, std, 'none', 'bilinear', order)
    np.testing.assert_allclose(shown, rgb / 255, atol=1e-7)
    pixels = rgb[0, 0] / 255
    if order == 'bgr':
        pixels = pixels[::-1]
    np.testing.assert_allclose(tensor[0, :, 0, 0].numpy(), (pixels - mean) / std, atol=1e-6)
    with pytest.raises(ValueError, match='channel_order'):
        preprocess(path, 4, mean, std, 'none', 'bilinear', 'invalid')


@pytest.mark.parametrize('text', ['["A", "B"]', 'A\n\nB\n', '  ["A", "B"]\n'])
def test_labels_accept_training_order(tmp_path, text):
    path = tmp_path / 'labels.txt'
    path.write_text(text, encoding='utf-8')
    assert load_labels(path, 2) == ['A', 'B']


@pytest.mark.parametrize('text', ['["A", 1]', '["A"]', '[broken', 'A\n'])
def test_labels_reject_invalid_input(tmp_path, text):
    path = tmp_path / 'labels.txt'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError):
        load_labels(path, 2)


def test_japanese_report_bgr_intervention_roundtrip(tmp_path):
    model = Toy().eval()
    x = sample()
    result = diagnose(model, x, 0)
    mean, std = np.array([.1, .2, .3]), np.array([.2, .3, .4])
    rgb = (x[0].permute(1, 2, 0).numpy() * std + mean)[:, :, ::-1]
    intervention = validate_regions(model, x, result, top_k=1)
    assert intervention[1]
    report = save_report(result, rgb, tmp_path / 'ja', class_names=['Correct prediction.', '<B>'],
                         intervention=intervention, normalization=(mean, std),
                         channel_order='bgr', language='ja')
    page = report.read_text(encoding='utf-8')
    assert '<html lang="ja">' in page
    assert 'ホット領域と低貢献領域の比較' in page
    assert '誤認識。' in page
    assert '<b>0 (Correct prediction.)</b>' in page
    assert '1 (&lt;B&gt;)' in page
    summary = json.loads((report.parent / 'summary.json').read_text())
    assert summary['contrast_class'] == summary['comparison_class'] == 1
    assert summary['is_correct'] is False
    assert summary['channel_order'] == 'bgr' and summary['language'] == 'ja'
    for key, (mask, tensor) in intervention[1].items():
        expected = np.clip(tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)[:, :, ::-1]
        with Image.open(report.parent / f'{key}.png') as image:
            np.testing.assert_array_equal(np.asarray(image), (expected * 255).round().astype(np.uint8))
        with Image.open(report.parent / f'{key}_mask.png') as image:
            np.testing.assert_array_equal(np.asarray(image), mask.astype(np.uint8)*255)
    data = json.loads((report.parent / 'interventions.json').read_text())
    assert data['contrast_class'] == data['comparison_class'] == 1
    assert all(row['contrast_class'] == row['comparison_class'] == 1
               for row in data['rows'] if row['status'] == 'ok')


def test_cli_image_options_end_to_end(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / 'weights.pth'
    model = Toy().eval()
    with torch.no_grad():
        model.classifier.bias.zero_()
    torch.save(model.state_dict(), checkpoint)
    image = tmp_path / 'input.png'
    Image.fromarray((sample()[0].permute(1, 2, 0).numpy()*255).astype(np.uint8)).save(image)
    labels = tmp_path / 'labels.txt'
    labels.write_text('A\nB\n', encoding='utf-8')
    output = tmp_path / 'report'
    monkeypatch.setattr(cli.timm, 'create_model', lambda *args, **kwargs: Toy())
    args = ['efficientnet-diagnose', '--checkpoint', str(checkpoint), '--image', str(image),
            '--true-class', '1', '--num-classes', '2', '--size', '4',
            '--mean', '0', '0', '0', '--std', '1', '1', '1', '--output', str(output),
            '--channel-order', 'bgr', '--language', 'ja', '--labels', str(labels)]
    monkeypatch.setattr(sys, 'argv', args)
    cli.main()
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['metadata']['channel_order'] == 'bgr'
    assert summary['metadata']['language'] == 'ja'
    # In BGR mode red moves to channel 2, so this Toy predicts A=0 instead of B=1.
    assert summary['predicted_class'] == 0
    assert '<html lang="ja">' in (output / 'index.html').read_text()
    assert 'misclassified' in capsys.readouterr().out
    labels.write_text('[broken', encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert 'error:' in capsys.readouterr().err

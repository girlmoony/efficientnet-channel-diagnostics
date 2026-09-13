"""Export channel tables, signed spatial overlays, and an offline HTML report."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from PIL import Image

from .analysis import Diagnosis
from .detail_report import save_details


def save_report(result: Diagnosis, input_rgb, output_dir, *, top_k=6,
                class_names=None, metadata=None, hot_fraction=.15, intervention=None,
                normalization=None):
    """RGB must match model input geometry, before normalization; no auto-resize.

    Use a NEW/empty output directory to avoid mixing reports. Heatmaps use
    signed values with constant opacity and a shared channel color scale.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < hot_fraction <= 1:
        raise ValueError("hot_fraction must be in (0,1]")
    if intervention is not None and normalization is None:
        raise ValueError("Supply normalization=(mean,std) to render intervention images")
    rgb = np.asarray(input_rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    if rgb.shape != (*result.input_size, 3) or not np.isfinite(rgb).all():
        raise ValueError("RGB image must match actual model input [H,W,3].")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB must be uint8 or floats in [0,1].")
    if class_names is not None and len(class_names) != len(result.logits):
        raise ValueError("Class names must match classifier output order and count.")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty (use a new report directory).")
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out / "input.png")
    d = result.margin_contribution.numpy()
    positive = np.flatnonzero(d > 0)
    negative = np.flatnonzero(d < 0)
    top_b = positive[np.argsort(-d[positive])][:top_k]
    top_a = negative[np.argsort(d[negative])][:top_k]

    fields = ["channel", "activation", "contribution_A", "contribution_B",
              "margin_contribution", "share_B", "share_A"]
    with (out / "channels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for k in range(len(d)):
            writer.writerow([k, float(result.activation[k]), float(result.contribution_a[k]),
                             float(result.contribution_b[k]), float(d[k]),
                             float(result.share_b[k]), float(result.share_a[k])])
    summary = {
        "true_class": result.true_class, "predicted_class": result.predicted_class,
        "comparison_class": result.comparison_class,
        "logit_A": float(result.logits[result.true_class]),
        "logit_B": float(result.logits[result.comparison_class]),
        "margin_B_minus_A": float(result.logits[result.comparison_class] - result.logits[result.true_class]),
        "bias_margin": result.bias_margin, "reconstruction_error": result.reconstruction_error,
        "channels": len(d), "feature_map_size": list(result.channel_maps.shape[-2:]),
        "input_size": list(result.input_size), "top_push_B": top_b.tolist(),
        "top_push_A": top_a.tolist(), "logits": result.logits.tolist(),
        "metadata": metadata or {},
        "hot_fraction": hot_fraction,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Original-resolution maps remain available for quantitative analysis.
    np.savez_compressed(out / "spatial_contributions.npz",
                        feature_maps=result.feature_maps.numpy(),
                        channel_maps=result.channel_maps.numpy(),
                        total_map=result.channel_maps.sum(0).numpy())

    fig, ax = plt.subplots(figsize=(15, 4), layout="constrained")
    ax.bar(np.arange(len(d)), d, width=1, color=np.where(d > 0, "tomato", "steelblue"))
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(xlabel="Channel (zero-based)", ylabel="Contribution to logit(B) - logit(A)",
           title="All channel contributions | red: pushes B, blue: pushes A")
    fig.savefig(out / "channel_distribution.png", dpi=160)
    plt.close(fig)

    def resize(m):
        return torch.nn.functional.interpolate(m[None, None], size=result.input_size,
                    mode="bilinear", align_corners=False)[0, 0].numpy()

    def overlay(ax, m, limit, title):
        ax.imshow(rgb)
        artist = ax.imshow(m, cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), alpha=.55)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return artist

    total = resize(result.channel_maps.sum(0))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("Actual model input")
    axes[0].axis("off")
    artist = overlay(axes[1], total, max(float(abs(total).max()), 1e-12), "Total B-vs-A spatial contribution")
    fig.colorbar(artist, ax=axes[1], shrink=.75, label="Blue: A | Red: B")
    fig.savefig(out / "total_regions.png", dpi=160)
    plt.close(fig)

    selected = np.concatenate([top_b, top_a])
    channel_images = {int(k): resize(result.channel_maps[int(k)]) for k in selected}
    limit = max([float(abs(m).max()) for m in channel_images.values()] + [1e-12])
    images = ["channel_distribution.png", "total_regions.png"]
    for ids, direction in [(top_b, "B"), (top_a, "A")]:
        if len(ids) == 0:
            continue
        cols = min(3, len(ids))
        rows = (len(ids) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False, layout="constrained")
        for index, channel in enumerate(ids):
            artist = overlay(axes.flat[index], channel_images[int(channel)], limit,
                             f"Channel {channel} | B-A: {d[channel]:+.4f}")
        for ax in list(axes.flat)[len(ids):]:
            ax.axis("off")
        fig.suptitle(f"Channels pushing {direction} | shared scale across both groups")
        fig.colorbar(artist, ax=list(axes.flat)[:len(ids)], shrink=.7)
        name = f"channels_push_{direction}.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        images.append(name)

    images += save_details(result, rgb, out, selected, hot_fraction)
    intervention_html = ""
    if intervention is not None:
        data, artifacts = intervention
        (out / "interventions.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        mean, std = (np.asarray(v).reshape(1, 1, 3) for v in normalization)
        for key, (mask, tensor) in artifacts.items():
            altered = np.clip(tensor.permute(1,2,0).numpy()*std + mean, 0, 1)
            Image.fromarray((altered*255).round().astype(np.uint8)).save(out / f"{key}.png")
            Image.fromarray(mask.astype(np.uint8)*255).save(out / f"{key}_mask.png")
        intervention_html = '<h2>Hot versus low-contribution region controls</h2><p>A (true class) and B (comparison class) remain fixed from the original input, even if Top-1 changes. Positive margin_drop means B-A decreased. Positive hot_minus_low_drop means hotspot replacement reduced B-A more than the control; a single result does not establish causality. <a href="interventions.json">Full experiment JSON</a></p>'
        intervention_html += '<table><tr><th>Target/method/region</th><th>Area fraction</th><th>B-A</th><th>Margin drop</th><th>New Top-1</th></tr>'
        for row in data["rows"]:
            if row["status"] == "ok":
                key = row["artifact"]
                intervention_html += f'<tr><td><a href="{key}.png">{key}</a> (<a href="{key}_mask.png">mask</a>)</td><td>{row["area_fraction"]:.3f}</td><td>{row["margin"]:.4f}</td><td>{row["margin_drop"]:.4f}</td><td>{row["predicted_class"]}</td></tr>'
            else:
                intervention_html += '<tr><td colspan="5">'+html.escape(str(row))+'</td></tr>'
        intervention_html += '</table>'

    def label(index):
        return html.escape(str(class_names[index])) if class_names is not None else str(index)

    outcome = "Correct" if result.predicted_class == result.true_class else "Incorrect"
    comparison_role = "runner-up" if outcome == "Correct" else "incorrect Top-1"
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EfficientNet Channel Diagnostics</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172338}}
img{{max-width:100%;height:auto}} .note{{background:#eef3f8;padding:18px;border-radius:8px}}
a{{color:#165ca3}} code{{background:#eee;padding:2px 5px}}</style>
<h1>Channel contributions and spatial diagnostics</h1>
<p>{outcome} prediction. True class A: <b>{label(result.true_class)}</b>;
predicted Top-1: <b>{label(result.predicted_class)}</b>;
comparison class B ({comparison_role}): <b>{label(result.comparison_class)}</b></p>
<p>B-A logit margin: {summary['margin_B_minus_A']:.6f}; bias difference: {result.bias_margin:.6f};
reconstruction error: {result.reconstruction_error:.2e}</p>
<div class="note">Red pushes B; blue supports A. Shares are computed within each direction,
exclude bias, and are not probabilities. Channel overlays share a color scale; the total map
uses its own scale. Native feature map size: {summary['feature_map_size']}; interpolation is for display only.
Spatial positions describe feature responses, not precise pixel-level causal attribution.
Background explanations require controlled image interventions.</div>
<p><a href="channels.csv">All channel values (CSV)</a> · <a href="summary.json">Scores and run configuration (JSON)</a> ·
<a href="spatial_contributions.npz">Native spatial contributions (NPZ)</a></p>'''
    page += "".join(f'<figure><img src="{name}" alt="{name}"><figcaption>{name}</figcaption></figure>' for name in images)
    page += '<p><a href="spatial_metrics.csv">All channel spatial concentration (CSV)</a> · <a href="spatial_metrics.json">Spatial metrics (JSON)</a></p><p>Metrics are computed separately for positive B contributions and absolute negative A contributions. Entropy is normalized by log(total grid cells); concentration shares and entropy are null when there is no contribution. Area coverage is the fraction of strictly positive grid cells, not object segmentation area. Detail panels use independent scales for each channel; compare numeric values.</p>'
    page += intervention_html + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"

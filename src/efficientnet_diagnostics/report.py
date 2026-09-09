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
        "logit_A": float(result.logits[result.true_class]),
        "logit_B": float(result.logits[result.predicted_class]),
        "margin_B_minus_A": float(result.logits[result.predicted_class] - result.logits[result.true_class]),
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
        intervention_html = '<h2>热区与低贡献区域对照</h2><p>A/B 固定为原预测比较对。正的 margin_drop 表示 B−A 降低。正的 hot_minus_low_drop 表示热区替换比对照影响大；单次结果不能证明因果。<a href="interventions.json">完整实验 JSON</a></p>'
        intervention_html += '<table><tr><th>目标/方法/区域</th><th>面积比例</th><th>B−A</th><th>下降量</th><th>新 Top-1</th></tr>'
        for row in data["rows"]:
            if row["status"] == "ok":
                key = row["artifact"]
                intervention_html += f'<tr><td><a href="{key}.png">{key}</a> (<a href="{key}_mask.png">mask</a>)</td><td>{row["area_fraction"]:.3f}</td><td>{row["margin"]:.4f}</td><td>{row["margin_drop"]:.4f}</td><td>{row["predicted_class"]}</td></tr>'
            else:
                intervention_html += '<tr><td colspan="5">'+html.escape(str(row))+'</td></tr>'
        intervention_html += '</table>'

    def label(index):
        return html.escape(str(class_names[index])) if class_names is not None else str(index)

    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EfficientNet 通道诊断</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172338}}
img{{max-width:100%;height:auto}} .note{{background:#eef3f8;padding:18px;border-radius:8px}}
a{{color:#165ca3}} code{{background:#eee;padding:2px 5px}}</style>
<h1>通道贡献与图片区域诊断</h1>
<p>真实 A：<b>{label(result.true_class)}</b>；预测 B：<b>{label(result.predicted_class)}</b></p>
<p>B−A logit：{summary['margin_B_minus_A']:.6f}；偏置差：{result.bias_margin:.6f}；
分解误差：{result.reconstruction_error:.2e}</p>
<div class="note">红色推动 B，蓝色支持 A。占比在同方向通道内计算，不包含偏置，也不是概率。
通道图共用色标，总图使用独立色标。原始特征图尺寸 {summary['feature_map_size']}，插值仅用于显示。
空间位置对应特征响应，不能视为精确的像素因果归因；背景原因需要受控图片干预验证。</div>
<p><a href="channels.csv">全部通道 CSV</a> · <a href="summary.json">分数与运行配置 JSON</a> ·
<a href="spatial_contributions.npz">原始空间贡献 NPZ</a></p>'''
    page += "".join(f'<figure><img src="{name}" alt="{name}"><figcaption>{name}</figcaption></figure>' for name in images)
    page += '<p><a href="spatial_metrics.csv">全部通道空间集中度 CSV</a> · <a href="spatial_metrics.json">空间指标 JSON</a></p><p>指标分别对 B 的正贡献和 A 的负贡献绝对值计算。熵按 log(全部格点数) 归一化；没有贡献时比例和熵为 null。覆盖面积为严格正贡献格点比例，不是物体分割面积。细节图使用各通道独立色标，应以数值比较。</p>'
    page += intervention_html + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"

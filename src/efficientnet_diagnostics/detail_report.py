"""Unsmoothed per-channel panels and spatial concentration tables."""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .spatial import hotspot, concentration, nearest


def save_details(result, rgb, out, selected, fraction):
    rows = []
    for k, m in enumerate(result.channel_maps.numpy()):
        row = {"channel": k}
        for direction, values in [("B", m), ("A", -m)]:
            row.update({f"{direction}_{key}": value for key, value in concentration(values).items()})
            row[f"{direction}_hot_grid_fraction"] = float(hotspot(values, fraction).mean())
        rows.append(row)
    with (out / "spatial_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "spatial_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    images = []
    for k in selected:
        k = int(k)
        m = result.channel_maps[k].numpy()
        activation = result.feature_maps[k].numpy()
        direction = "B" if result.margin_contribution[k] > 0 else "A"
        values = m if direction == "B" else -m
        mask = hotspot(values, fraction)
        limit = max(float(abs(m).max()), 1e-12)
        norm = TwoSlopeNorm(0, -limit, limit)
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), layout="constrained")
        artist = axes[0].imshow(m, cmap="RdBu_r", norm=norm, interpolation="nearest")
        for y, x in np.ndindex(m.shape):
            axes[0].text(x, y, f"{m[y,x]:.2g}", ha="center", va="center", fontsize=7,
                         color="white" if abs(m[y,x]) > .6*limit else "black")
        axes[0].set_title(f"Native signed contribution {m.shape}")
        axes[0].set_xticks(range(m.shape[1]))
        axes[0].set_yticks(range(m.shape[0]))
        axes[1].imshow(nearest(m, result.input_size), cmap="RdBu_r", norm=norm, interpolation="nearest")
        axes[1].set_title("Nearest blocks (no RGB blending)")
        fig.colorbar(artist, ax=list(axes[:2]), shrink=.65)
        gray = axes[2].imshow(activation, cmap="gray", interpolation="nearest")
        axes[2].set_title("Actual activation (own grayscale)")
        fig.colorbar(gray, ax=axes[2], shrink=.65)
        axes[3].imshow(rgb)
        # Draw exact exterior edges on the nearest-resized binary mask, including borders.
        up = nearest(mask, result.input_size).astype(bool)
        from matplotlib.collections import LineCollection
        edges = []
        for y, x in zip(*np.where(up)):
            if y == 0 or not up[y-1,x]: edges.append([(x-.5,y-.5),(x+.5,y-.5)])
            if y == up.shape[0]-1 or not up[y+1,x]: edges.append([(x-.5,y+.5),(x+.5,y+.5)])
            if x == 0 or not up[y,x-1]: edges.append([(x-.5,y-.5),(x-.5,y+.5)])
            if x == up.shape[1]-1 or not up[y,x+1]: edges.append([(x+.5,y-.5),(x+.5,y+.5)])
        axes[3].add_collection(LineCollection(edges, colors="lime", linewidths=1.4))
        axes[3].set_title(f"{direction} hotspot: {mask.sum()}/{mask.size} cells\nTop {fraction:.0%} positive cells; ties included")
        for ax in axes[1:]: ax.axis("off")
        metrics = concentration(values)
        fig.suptitle(f"Channel {k} | supports {direction} | concentration: {metrics}", fontsize=10)
        name = f"channel_{k}_details.png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        images.append(name)
    return images

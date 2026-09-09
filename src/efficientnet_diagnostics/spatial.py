"""Native-grid statistics and explicit, signed-direction hotspot selection."""
import numpy as np
import torch


def hotspot(values, fraction=.15):
    """Top fraction of strictly positive cells, including threshold ties."""
    if not 0 < fraction <= 1:
        raise ValueError("hot fraction must be in (0,1]")
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if not positive.size:
        return np.zeros(values.shape, dtype=bool)
    return (values > 0) & (values >= np.quantile(positive, 1 - fraction))


def concentration(values):
    """Statistics of positive mass; entropy normalized by log(all grid cells)."""
    v = np.maximum(np.asarray(values, dtype=float), 0).ravel()
    total = v.sum()
    if total == 0:
        return dict(max_share=None, top4_share=None, entropy=None, positive_area=0.)
    p = v[v > 0] / total
    entropy = float(-(p * np.log(p)).sum() / np.log(v.size)) if v.size > 1 else 0.
    return dict(max_share=float(v.max()/total), top4_share=float(np.sort(v)[-4:].sum()/total),
                entropy=entropy, positive_area=float(np.mean(v > 0)))


def nearest(values, size):
    tensor = torch.as_tensor(np.asarray(values).copy(), dtype=torch.float32)
    return torch.nn.functional.interpolate(tensor[None, None], size=size,
                                           mode="nearest")[0, 0].numpy()

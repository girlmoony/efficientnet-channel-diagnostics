"""Fixed A/B, equal-area hot versus low-region replacement experiments."""
import numpy as np
import torch
from .spatial import hotspot, nearest


@torch.no_grad()
def validate_regions(model, x, result, *, fraction=.15, top_k=3):
    """Replace normalized pixels with image-channel mean or local average blur.
    Both baselines are affine-compatible with standard per-channel normalization.
    Returns data plus masks and modified normalized tensors for audit.
    """
    if any(m.training for m in model.modules()):
        raise ValueError("Call model.eval() first")
    if tuple(x.shape) != (1, 3, *result.input_size):
        raise ValueError("Input shape does not match diagnosis")
    original = model(x)[0]
    torch.testing.assert_close(original.cpu(), result.logits, rtol=1e-4, atol=1e-5)
    a, b = result.true_class, result.predicted_class
    baseline = float(original[b] - original[a])
    maps = {"total": result.channel_maps.sum(0).numpy()}
    ids = torch.argsort(result.margin_contribution, descending=True)
    ids = ids[result.margin_contribution[ids] > 0][:top_k]
    maps.update({f"channel_{int(k)}": result.channel_maps[k].numpy() for k in ids})
    mean = x.mean((2, 3), keepdim=True).expand_as(x)
    # Replicate padding avoids artificial dark borders.
    kernel = min(31, min(result.input_size))
    kernel = kernel if kernel % 2 else kernel - 1
    blur = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(
        x, (kernel//2,)*4, mode="replicate"), kernel, stride=1)
    rows, artifacts = [], {}
    for name, values in maps.items():
        hot = nearest(hotspot(values, fraction), result.input_size).astype(bool)
        count = int(hot.sum())
        available = np.flatnonzero(~hot.ravel())
        if not count or len(available) < count:
            rows.append(dict(target=name, status="skipped", reason="No positive hotspot or insufficient disjoint control area (ties may cover whole grid)."))
            continue
        # Exact equal pixel area even for input sizes not divisible by grid size.
        # Prefer lowest nonnegative B evidence outside the hot mask; stable ties.
        score = nearest(np.maximum(values, 0), result.input_size).ravel()
        chosen = available[np.argsort(score[available], kind="stable")[:count]]
        low = np.zeros(hot.size, dtype=bool)
        low[chosen] = True
        low = low.reshape(hot.shape)
        for method, replacement in [("mean", mean), ("blur", blur)]:
            drops = {}
            for region, mask in [("hot", hot), ("low", low)]:
                mask_t = torch.as_tensor(mask, device=x.device)[None, None]
                modified = torch.where(mask_t, replacement, x)
                logits = model(modified)[0]
                margin = float(logits[b] - logits[a])
                key = f"{name}_{method}_{region}"
                rows.append(dict(target=name, status="ok", method=method, region=region,
                                 pixels=count, area_fraction=count/hot.size,
                                 logit_A=float(logits[a]), logit_B=float(logits[b]),
                                 margin=margin, margin_drop=baseline-margin,
                                 predicted_class=int(logits.argmax()), artifact=key))
                drops[region] = baseline-margin
                artifacts[key] = (mask, modified[0].cpu())
            rows.append(dict(target=name, status="comparison", method=method,
                             hot_minus_low_drop=drops["hot"]-drops["low"]))
    return dict(original_margin=baseline, A=a, B=b, fraction=fraction, rows=rows), artifacts

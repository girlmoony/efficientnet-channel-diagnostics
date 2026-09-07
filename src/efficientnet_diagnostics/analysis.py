"""Contrastive attribution for an unmodified timm GAP + Linear head."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Diagnosis:
    true_class: int
    predicted_class: int
    logits: torch.Tensor
    activation: torch.Tensor
    contribution_a: torch.Tensor
    contribution_b: torch.Tensor
    margin_contribution: torch.Tensor
    share_b: torch.Tensor
    share_a: torch.Tensor
    channel_maps: torch.Tensor
    bias_margin: float
    input_size: tuple[int, int]
    reconstruction_error: float


@torch.no_grad()
def diagnose(model: nn.Module, x: torch.Tensor, true_class: int) -> Diagnosis:
    """Analyze one FP32 image. Keeps model mode unchanged; call model.eval() first.

    Only standard single-image FP32, average-pool + Linear classifiers are
    supported. Validate against the actual forward pass, including head hooks.
    Returned tensors are detached CPU tensors; no gradients are required.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Call model.eval() before diagnosis (Dropout/BN must be disabled).")
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
        raise ValueError("Expected one RGB image [1, 3, H, W].")
    if x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("Input must be finite FP32; disable autocast for diagnosis.")
    fc = getattr(model, "classifier", None)
    if not isinstance(fc, nn.Linear) or fc.weight.dtype != torch.float32:
        raise ValueError("Expected an FP32 model.classifier Linear layer.")
    a = int(true_class)
    if a != true_class or not 0 <= a < fc.out_features:
        raise ValueError("true_class must be a valid zero-based class index.")

    captured = {}

    def capture_pool(module, args):
        captured["maps"] = args[0].detach().clone()

    def capture_classifier(module, args):
        captured["pooled"] = args[0].detach().clone()

    pool = getattr(model, "global_pool", None)
    if pool is None:
        raise ValueError("Expected model.global_pool.")
    handles = [pool.register_forward_pre_hook(capture_pool),
               fc.register_forward_pre_hook(capture_classifier)]
    try:
        output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    maps, pooled = captured["maps"], captured["pooled"]
    if not isinstance(output, torch.Tensor) or output.shape != (1, fc.out_features):
        raise ValueError("Model must return raw logits [1, num_classes].")
    if maps.ndim != 4 or pooled.shape != (1, fc.in_features):
        raise ValueError("Unsupported feature dimensions.")
    if maps.dtype != torch.float32 or pooled.dtype != torch.float32:
        raise ValueError("Disable autocast and use model.float().")
    if not all(torch.isfinite(t).all() for t in (maps, pooled, output, fc.weight)):
        raise ValueError("Non-finite features, weights or logits.")
    # These checks reject max pooling, extra nonlinear heads, output softmax, etc.
    torch.testing.assert_close(pooled, maps.mean((2, 3)), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(output, torch.nn.functional.linear(
        pooled, fc.weight, fc.bias), rtol=1e-4, atol=1e-5)

    logits = output[0]
    b = int(logits.argmax())
    if a == b:
        raise ValueError("Prediction equals true_class; no A-to-B misclassification.")
    h = pooled[0]
    ca, cb = h * fc.weight[a], h * fc.weight[b]
    delta = cb - ca
    bias = (fc.bias[b] - fc.bias[a]) if fc.bias is not None else h.new_zeros(())
    spatial = maps[0] * (fc.weight[b] - fc.weight[a])[:, None, None]
    torch.testing.assert_close(spatial.mean((1, 2)), delta, rtol=1e-4, atol=1e-4)
    reconstructed = delta.sum() + bias
    torch.testing.assert_close(reconstructed, logits[b] - logits[a], rtol=1e-4, atol=1e-4)
    positive, negative = delta.clamp_min(0), (-delta).clamp_min(0)

    return Diagnosis(
        a, b, logits.cpu(), h.cpu(), ca.cpu(), cb.cpu(), delta.cpu(),
        (positive / positive.sum().clamp_min(1e-12)).cpu(),
        (negative / negative.sum().clamp_min(1e-12)).cpu(), spatial.cpu(),
        float(bias), tuple(x.shape[-2:]),
        float((reconstructed - (logits[b] - logits[a])).abs()),
    )

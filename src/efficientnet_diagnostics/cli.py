"""Command line interface. Preprocessing is explicit, never silently inferred."""
import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch

from .analysis import diagnose
from .report import save_report
from .intervention import validate_regions


def load_weights(model, path, key=None):
    """Load tensor state dictionaries only, strictly; never unpickle a model."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if key:
        state = state[key]
    elif isinstance(state, Mapping):
        for candidate in ("state_dict", "model_state_dict"):
            if candidate in state:
                state = state[candidate]
                break
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Expected a tensor state_dict; select nested weights with --checkpoint-key.")
    # Standard DistributedDataParallel checkpoint prefix only.
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def preprocess(path, size, mean, std, resize, interpolation):
    if not all(np.isfinite(mean)) or not all(np.isfinite(std)) or min(std) <= 0:
        raise ValueError("mean/std must be finite and std must be positive")
    with Image.open(path) as source:
        if source.mode != "RGB":
            raise ValueError("CLI requires RGB input. For alpha/grayscale use your existing preprocessing via Python API.")
        rgb = source.copy()
    if resize == "none" and rgb.size != (size, size):
        raise ValueError("Image size differs from --size; explicitly select --resize stretch or use Python API.")
    if resize == "stretch":
        methods = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
                   "nearest": Image.Resampling.NEAREST}
        rgb = rgb.resize((size, size), methods[interpolation])
    array = np.asarray(rgb).astype(np.float32) / 255
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    return array, tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Diagnose A-to-B channel and spatial contributions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-key")
    parser.add_argument("--image", required=True)
    parser.add_argument("--true-class", required=True, type=int)
    parser.add_argument("--model", default="tf_efficientnet_lite1")
    parser.add_argument("--num-classes", type=int, default=400)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--mean", type=float, nargs=3, required=True)
    parser.add_argument("--std", type=float, nargs=3, required=True)
    parser.add_argument("--resize", choices=["none", "stretch"], default="none")
    parser.add_argument("--interpolation", choices=["bilinear", "bicubic", "nearest"], default="bilinear")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hot-fraction", type=float, default=.15)
    parser.add_argument("--validate-regions", action="store_true", help="Mean/blur hot-vs-low input replacement")
    parser.add_argument("--validation-top-k", type=int, default=3)
    parser.add_argument("--labels", help="UTF-8 JSON array of labels in training index order")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.hot_fraction <= 1 or args.validation_top_k < 0:
        parser.error("hot-fraction must be in (0,1]; validation-top-k must be nonnegative")
    if args.top_k < 1 or args.size < 1 or args.num_classes < 2:
        parser.error("top-k/size must be positive and num-classes at least 2")
    names = None
    if args.labels:
        names = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        if not isinstance(names, list) or len(names) != args.num_classes or not all(isinstance(n, str) for n in names):
            parser.error("labels must be a JSON array with num-classes strings")
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    load_weights(model, args.checkpoint, args.checkpoint_key)
    model = model.float().to(args.device).eval()
    rgb, x = preprocess(args.image, args.size, args.mean, args.std, args.resize, args.interpolation)
    result = diagnose(model, x.to(args.device), args.true_class)
    intervention = validate_regions(model, x.to(args.device), result,
        fraction=args.hot_fraction, top_k=args.validation_top_k) if args.validate_regions else None
    metadata = {"model": args.model, "mean": args.mean, "std": args.std,
                "resize": args.resize, "interpolation": args.interpolation,
                "torch_version": str(torch.__version__), "timm_version": timm.__version__}
    report = save_report(result, rgb, args.output, top_k=args.top_k,
                         class_names=names, metadata=metadata, hot_fraction=args.hot_fraction,
                         intervention=intervention, normalization=(args.mean, args.std))
    print(f"A={result.true_class}, B={result.predicted_class}")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()

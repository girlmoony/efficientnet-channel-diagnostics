# EfficientNet Channel Diagnostics

Contrastive channel and spatial diagnostics for PyTorch / timm `tf_efficientnet_lite1`, for both correct and incorrect predictions. No retraining or gradients are required.

## Class selection

A is always the supplied true class. `predicted_class` is the actual model Top-1. `comparison_class` is B:

- **Correct prediction:** A is the true/predicted Top-1; B is the highest-logit class other than A (the runner-up).
- **Incorrect prediction:** B is the incorrect Top-1, preserving the original comparison.

Ties select the lowest class index, following PyTorch `argmax`. At least two classes are required. All contributions describe **B minus A**: a positive value pushes B relative to A; a negative value supports A. The margin is nonpositive for correct predictions and nonnegative for incorrect predictions, including zero for ties.

Both cases use the same complete pipeline: channel contributions, spatial maps, channel detail panels, hotspots, spatial metrics, and optional intervention outputs. Directional plots are generated when contributions in that direction exist; unavailable interventions record a skipped reason.

The default model has 400 classes and takes 256x256 RGB input. Supply your trained weights; this repository includes neither training data nor sushi model weights.

## Installation

Python 3.10+ is required. Create a virtual environment and install PyTorch for your hardware. CPU example:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## Command line

The mean/std below are **examples only**. Replace them with the values used during your actual inference; do not infer preprocessing from the model name.

```bash
efficientnet-diagnose --checkpoint checkpoints/sushi.pth --image data/image.png --true-class 12 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --output reports/case-001
```

`--true-class 12` is the zero-based training index of the true class A. B is selected automatically using the rule above. Correct predictions generate a report rather than exiting. CLI output identifies the true class A, `predicted_class`, and `comparison_class` B separately.

| Option | Default / meaning |
|---|---|
| `--checkpoint` | Required trained state dictionary |
| `--checkpoint-key` | Optional key containing the state dictionary |
| `--image` | Required RGB image |
| `--true-class` | Required zero-based true class index |
| `--model` | `tf_efficientnet_lite1` |
| `--num-classes` | `400`; must match the checkpoint |
| `--size` | `256` |
| `--mean`, `--std` | Required three-channel normalization values |
| `--resize` | `none` requires the exact input size; `stretch` explicitly resizes to a square |
| `--interpolation` | `bilinear`; also `bicubic` or `nearest`, used only with stretch |
| `--device` | `cpu`; `cuda` is supported |
| `--top-k` | Show 6 channels per direction |
| `--hot-fraction` | `0.15`; fraction of strictly positive directional grid cells selected by quantile, retaining ties |
| `--validate-regions` | Enable mean/blur hotspot-versus-control replacements |
| `--validation-top-k` | `3`; number of B-supporting channels to validate, in addition to the total map |
| `--labels` | Optional UTF-8 JSON string array in training class order |
| `--output` | Required new/empty report directory to avoid mixing cases |

Checkpoint loading accepts a plain state dictionary, `state_dict` / `model_state_dict` wrappers, and a uniform `module.` prefix. All weights load strictly with `weights_only=True`; serialized full model objects are not loaded and missing weights are not silently ignored.

The CLI does not apply EXIF orientation, center cropping, or alpha compositing. RGB conversion, resizing, and normalization must match original inference. Use the Python API for custom preprocessing, crops, or transparent images.

## Python API

```python
from efficientnet_diagnostics import diagnose, save_report

# model: timm model with your trained weights loaded
# x: original inference tensor [1, 3, 256, 256], with identical preprocessing
# input_rgb: aligned [256, 256, 3] RGB after resize/crop, before normalization
model.float().eval()
result = diagnose(model, x.float(), true_class=A_index)
print(result.predicted_class, result.comparison_class)
path = save_report(result, input_rgb, "reports/case-002", top_k=6)
print(path)
```

Keep input and model on the same device. Disable autocast and use FP32. The model must have standard `global_pool` average pooling and a Linear `classifier`, and return raw logits. Hooks capture the pre-pool feature maps and classifier input during one actual forward pass, then are removed. The implementation does not depend on specific BN or activation layer names.

If you have only a tensor normalized with `(rgb - mean) / std`, recover aligned RGB as follows:

```python
mean_t = x.new_tensor(mean).view(3, 1, 1)
std_t = x.new_tensor(std).view(3, 1, 1)
input_rgb = ((x[0].detach() * std_t + mean_t)
             .clamp(0, 1).permute(1, 2, 0).cpu().numpy())
```

Do not overlay features from a center crop onto the uncropped original. Reports use actual model input coordinates; mapping back to original image coordinates is outside this version's scope.

## Channel detail panels and spatial metrics

Each displayed channel has a `channel_<id>_details.png` containing:

1. A native signed contribution grid with numeric cell values and no smoothing. Lite1 typically produces an 8x8 grid; the actual shape is reported.
2. Nearest-neighbor blocks preserving grid structure without RGB blending.
3. The actual pre-pool activation map, without classifier weights, using an independent grayscale scale.
4. Hotspot boundaries on the actual model input. B-supporting channels use positive contributions; A-supporting channels use absolute negative contributions.

`--hot-fraction 0.15` selects the top 15% of strictly positive cells in the chosen direction using a quantile threshold. Values such as `0.10` or `0.20` are also supported. All threshold ties are retained, so the actual selected fraction can exceed the requested fraction. Panels show the actual cell count. Uniform responses can select the entire grid and should not be interpreted as precise localization.

`spatial_metrics.csv/json` contains separate A/B directional metrics for every channel:

| Metric | Meaning |
|---|---|
| `max_share` | Largest cell's share of the directional contribution total |
| `top4_share` | Top four cells' share of the directional total |
| `entropy` | Shannon entropy divided by log(total grid cells); 0 means concentrated and 1 means uniform over the entire grid |
| `positive_area` | Fraction of all cells with strictly positive directional contribution, without a noise threshold |
| `hot_grid_fraction` | Actual fraction of hotspot cells, including threshold ties |

Concentration shares and entropy are null when no directional contribution exists. Coverage describes the feature grid, not pixel importance or object area. Raw activations are also stored as `feature_maps` in `spatial_contributions.npz`. Detail panels use independent scales per channel with numeric values/colorbars; A/B channel overlay groups share a scale.

## Optional region replacement experiments

These experiments require additional inference passes:

```bash
efficientnet-diagnose --checkpoint checkpoints/sushi.pth --image data/image.png --true-class 12 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --output reports/case-003 --hot-fraction 0.15 --validate-regions --validation-top-k 3
```

Replace mean/std with your actual configuration. Experiments cover the total B-A spatial map and the top three channels pushing B. **A and B stay fixed from the original diagnosis**, including when B is the runner-up for a correct prediction or when an intervention changes Top-1.

- Map hotspots to input pixels with nearest-neighbor resizing and select a disjoint low-contribution control with exactly the same pixel area.
- Select the control outside the hotspot in ascending order of `max(B-A contribution, 0)`. Negative contributions count as low B evidence. Ties use row-major order; grid cells may be partially selected to match the exact area. This is a deterministic control, not a randomized trial.
- Replace hot and low regions separately with the whole image's per-channel mean or a local average blur. The blur kernel is at most 31 and shrinks for small inputs. Both replacements operate in standard per-channel normalized space and are equivalent to the same linear operations in RGB space.
- Record fixed A/B logits, margin, `margin_drop = original margin - new margin`, the new full-classifier Top-1, and the `hot_minus_low_drop` control difference.
- Record a skipped reason if there is no positive hotspot or if ties make the hotspot too large for a disjoint equal-area control.

The HTML includes results, modified images, and binary masks; `interventions.json` contains complete data. Its top-level `predicted_class` is the original Top-1 and `comparison_class` is fixed B. Successful rows contain the modified image's `predicted_class` and the same fixed `comparison_class`. The existing `A` and `B` keys remain available.

A positive `hot_minus_low_drop` means hotspot replacement weakened B relative to A more than the control did. Negative differences are retained. Agreement between mean and blur is stronger evidence, but replacement artifacts, receptive fields, and control selection still affect the result. This does not prove background shortcuts in the training data.

```python
from efficientnet_diagnostics import diagnose, validate_regions, save_report

result = diagnose(model, x, A_index)
experiment = validate_regions(model, x, result, fraction=.15, top_k=3)
save_report(result, input_rgb, "reports/case-004", hot_fraction=.15,
            intervention=experiment, normalization=(mean, std))
```

## Outputs

Open `reports/case-001/index.html` without a server. Share the entire report directory, since HTML references adjacent images.

| File | Contents |
|---|---|
| `index.html` | English report with prediction outcome and separate true, predicted, and comparison classes |
| `channels.csv` | All channel values in zero-based channel order |
| `channel_distribution.png` | Signed contribution bars for all channels |
| `total_regions.png` | Actual model input and total spatial contribution |
| `channels_push_B.png` | Top B-supporting channel overlays, when positive contributions exist |
| `channels_push_A.png` | Top A-supporting channel overlays, when negative contributions exist |
| `channel_<id>_details.png` | Native grid, nearest blocks, activation, and hotspot boundaries for each selected channel |
| `spatial_metrics.csv/json` | Directional spatial metrics for every channel |
| `input.png` | Model input aligned with heatmaps |
| `summary.json` | `true_class`, actual `predicted_class`, `comparison_class` B, A/B logits, margin, bias, top channels, all logits, and run configuration |
| `spatial_contributions.npz` | Native `[C,Hf,Wf]` channel maps, total map, and raw feature maps |
| `interventions.json` | Optional fixed-pair experiment results and skipped reasons |
| `<target>_<method>_<region>.png`, `*_mask.png` | Optional modified inputs and binary masks |

`logit_B` and `margin_B_minus_A` always use `comparison_class`, which differs from `predicted_class` on correct predictions. Consumers that previously treated `predicted_class` as B should use `comparison_class` instead.

## Mathematical interpretation

For pre-pool features `F[k,u,v]`, pooled activations `h[k] = mean(F[k])`, and classifier weights `W[class,k]`:

```text
Contribution to A: C_A[k] = W[A,k] * h[k]
Contribution to B: C_B[k] = W[B,k] * h[k]
Contrastive contribution: D[k] = (W[B,k] - W[A,k]) * h[k]
Spatial contribution: M[k,u,v] = (W[B,k] - W[A,k]) * F[k,u,v]

mean(M[k]) = D[k]
sum(D) + bias[B] - bias[A] = logit[B] - logit[A]
```

Both identities are checked automatically for either prediction outcome. Spatial **means**, not sums, equal channel contributions. Divide by `Hf*Wf` if you need spatial sums to equal channel contributions.

`share_B[k] = max(D[k],0) / sum(max(D,0))`. A shares use absolute negative contributions in the same way. Shares are zero when no contribution in that direction exists. They exclude bias and are neither class probabilities nor misclassification probabilities.

The total heatmap excludes bias because bias has no spatial location. Total and per-channel maps use different scales; both directional channel overlay groups share a scale to preserve strength comparisons.

## Scope and limitations

- Exactly decomposes the final linear classifier's B-A score difference, whether A or B wins.
- Shows the corresponding spatial distribution in the final feature maps.
- A standard 256 input usually gives an 8x8 final grid. Interpolating to 256x256 adds no localization precision.
- Final features have large receptive fields. A hotspot on the background does not prove that only background pixels were used.
- A channel pushing B is not inherently a bad channel and does not necessarily encode background. Background shortcuts require training-sample controls and image interventions.
- Quantized or mixed-precision attribution, nonlinear heads, output softmax, multi-image batches, and arbitrary model architectures are unsupported.
- Integrated Gradients, Grad-CAM/LayerCAM, and earlier-layer attribution are not implemented. Final-layer single-channel maps are not precise object/pixel localization.

## Validation

```bash
python -m pytest -q
```

Tests cover known spatial contributions, both prediction outcomes, runner-up selection and ties, signed shares, bias and no bias, margin reconstruction, channel ablation, invalid pooling and class counts, complete reports and interventions, input alignment, checkpoint loading, CLI output, and the real timm `tf_efficientnet_lite1` 1280x8x8 structure.

The real-structure test uses randomly initialized weights to validate computation and interfaces; it is not a diagnosis of your trained model. Validate preprocessing and prediction consistency with your own weights and samples.

Reference: [timm EfficientNet source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py).

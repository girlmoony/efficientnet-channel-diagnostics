# EfficientNet Channel Diagnostics

针对 PyTorch / timm `tf_efficientnet_lite1` 的 **A→B 误分类分析**：找出推动错误类别 B 的通道，并把贡献映射到模型实际输入图片的区域。

## 两个功能

1. **通道贡献分布**：导出全部 1280 通道的激活、对 A/B 的贡献、B−A 贡献、同方向占比，以及贡献柱状图。
2. **对应图片区域**：输出总空间贡献图，以及最推动 B、最支持 A 的通道叠加图。红色推动 B，蓝色支持 A；各通道共用色标。

无需重新训练，不需要梯度。默认模型为 400 类、256×256 RGB 输入。使用你自己的训练权重；仓库不包含训练数据或寿司模型权重。

## 安装

Python 3.10+。建议新建虚拟环境，先按自己的硬件安装 PyTorch；下面是 CPU 示例：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## 命令行使用

下面的 mean/std **仅作示例**。必须替换成你实际推理采用的配置；不能仅根据模型名称猜测。

```bash
efficientnet-diagnose --checkpoint checkpoints/sushi.pth --image data/error.png --true-class 12 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --output reports/case-001
```

`--true-class 12` 是真实类别 A 的训练索引，**从 0 开始**。错误类别 B 自动取实际模型的 400 类 Top-1，不是手动指定。预测正确时程序会明确退出。

支持以下选项：

| 参数 | 默认值 / 说明 |
|---|---|
| `--model` | `tf_efficientnet_lite1` |
| `--num-classes` | `400`，必须和 checkpoint 一致 |
| `--size` | `256` |
| `--resize` | `none`；默认要求图片尺寸已经正确，`stretch` 显式缩放至正方形 |
| `--interpolation` | `bilinear`，仅 stretch 时生效；可选 bicubic/nearest |
| `--device` | `cpu`；可用 `cuda` |
| `--top-k` | 每个方向展示 6 个通道 |
| `--labels` | 可选 JSON 字符串数组，顺序必须与训练一致 |
| `--checkpoint-key` | 可选，指定 checkpoint 内存储 state_dict 的键 |
| `--output` | 必填，新的/空的报告目录，避免混合不同图片结果 |

接受纯 state_dict、`state_dict`/`model_state_dict` 包装，以及统一 `module.` 前缀。严格加载全部权重，使用 `weights_only=True`，不加载序列化完整模型对象，不静默忽略缺失权重。

CLI 不自动应用 EXIF 转向、中心裁剪或透明通道合成。RGB/缩放/归一化必须与原推理一致。若你的流程包含裁剪、透明 PNG、特殊图像处理，使用下面的 Python API 接入现有预处理。

## Python API：接入现有模型和预处理

```python
from efficientnet_diagnostics import diagnose, save_report

# model: 已加载你训练权重的 timm 模型
# x: 你原推理使用的 [1,3,256,256] 张量，含相同缩放/裁剪/归一化
# input_rgb: 相同缩放/裁剪之后、Normalize 之前的 [256,256,3] RGB
model.float().eval()
result = diagnose(model, x.float(), true_class=A_index)
path = save_report(result, input_rgb, "reports/case-002", top_k=6)
print(path)
```

输入和模型放在同一设备。关闭 autocast；分析使用 FP32。模型必须是标准 `global_pool` 平均池化和 `classifier` 线性分类头，返回原始 logits。API 在一次真实 forward 中通过 hook 捕获池化前空间特征和分类器输入，之后清理 hook。不会依赖易变的 BN/激活层名称。

如果只有归一化张量，且使用 `(rgb - mean) / std`：

```python
mean_t = x.new_tensor(mean).view(3, 1, 1)
std_t = x.new_tensor(std).view(3, 1, 1)
input_rgb = ((x[0].detach() * std_t + mean_t)
             .clamp(0, 1).permute(1, 2, 0).cpu().numpy())
```

不要把中心裁剪后的特征叠加到未裁剪原图。报告明确使用模型实际输入坐标，原图坐标反映射不在本版本范围内。

## 输出

打开 `reports/case-001/index.html`，无需启动服务器。报告引用同目录图片；分享时请打包整个报告目录。

| 文件 | 内容 |
|---|---|
| `channels.csv` | 全部通道数值，按 0-based 通道编号排列 |
| `channel_distribution.png` | 全部通道正负贡献柱状图 |
| `total_regions.png` | 模型实际输入与总空间贡献图 |
| `channels_push_B.png` | 最推动 B 的通道区域图，有正贡献时输出 |
| `channels_push_A.png` | 最支持 A 的通道区域图，有负贡献时输出 |
| `input.png` | 与热图对齐的模型输入 |
| `summary.json` | A/B logits、偏置、Top 通道、全部类别 logits、运行配置 |
| `spatial_contributions.npz` | 原分辨率全部通道图 `[C,Hf,Wf]` 和总图 |

## 数学含义

设池化前特征为 `F[k,u,v]`，`h[k] = mean(F[k])`，分类权重为 `W[class,k]`：

```text
对 A 的贡献：C_A[k] = W[A,k] * h[k]
对 B 的贡献：C_B[k] = W[B,k] * h[k]
误判方向贡献：D[k] = (W[B,k] - W[A,k]) * h[k]
空间贡献：M[k,u,v] = (W[B,k] - W[A,k]) * F[k,u,v]

mean(M[k]) = D[k]
sum(D) + bias[B] - bias[A] = logit[B] - logit[A]
```

这两个等式会自动验证。通道的空间图是**空间平均**等于通道贡献；若需要空间求和等于贡献，应再除以 `Hf*Wf`。

`share_B[k] = max(D[k],0) / sum(max(D,0))`；支持 A 的占比对负贡献的绝对值同样计算。没有对应方向的贡献时占比均为 0。占比不含偏置，不是类别概率，也不是误判概率。

总热图不包含偏置，因为偏置不具有空间位置。总热图和单通道图使用不同色标；单通道的 A/B 两组共用色标，避免逐图归一化造成强弱误判。

## 能解释什么，以及限制

- 可以精确分解**最终线性分类头**为何给 B 比 A 更高的分数。
- 可以展示该分数差对应的最后特征图空间分布。
- 标准 256 输入下最后空间图通常为 8×8，以运行结果为准。插值到 256×256 不会增加定位精度。
- 最后特征的感受野较大，热图落在背景并不证明只使用了背景像素。
- “通道 317 推动了 B”不等于“通道 317 是坏通道”，也不证明它代表背景概念。背景伪相关仍需训练样本对照和图片干预验证。
- 不支持量化/混合精度归因、非线性分类头、输出 softmax、多图 batch 或任意模型结构。

## 验证

```bash
python -m pytest -q
```

测试包含已知左右区域贡献、正负占比、偏置与无偏置、通道消融的分数变化、错误池化拒绝、报告完整性、输入对齐、checkpoint 加载，以及真实 timm `tf_efficientnet_lite1` 的 1280×8×8 结构验证。

真实结构测试使用随机初始化权重，只验证计算和接口，不代表你的寿司模型诊断结果。尚需用你自己的模型/样本验证实际预处理和预测一致性。

参考：[timm EfficientNet 源码](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py)。

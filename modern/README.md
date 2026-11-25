# Pan-Mamba — release-compatible modernization

Pinned reference: [original public commit ec1b9ea](https://github.com/alexhe101/Pan-Mamba/tree/ec1b9ead9a02b482851be98b7783111880bdd176). This is an implementation refactor, not a paper-inspired redesign.

## Compatibility contract

- Preserve the original fixed 32-channel architecture, all parameter and buffer names (including unused registered tensors), the 8+8 independent Mamba blocks, 2 swap blocks and 5 cross-modal blocks.
- Preserve **nearest** 4× upsampling (the historical variable is named `ms_bic`), double normalization, sequential MS/PAN shallow fusion, both state-space branches and residual order.
- Split the effective layers and Mamba blocks into modules. Ship the exact released fork from `Mamba-block/mamba_simple.py` locally; only two invalid optional-import assignments are corrected. No changes to the SSM equations or fused kernels.
- Pass actual height and width to the cross-modal reshape. On the originally hardcoded 1024×1024 grid this is the same view; smaller/non-square inputs are an explicit shape-compatibility extension, not a different fusion algorithm.
- Missing CUDA kernels raise an error; no silent substitute, no Mamba-2, no changing `use_fast_path`. CPU-only test recurrence is never used by production.

The three original .pth files are kept byte-for-byte in their original paths. The loader supports the released `net` wrapper and DataParallel `module.` prefix and uses `strict=True`. No parameter migration, padding, truncation or `strict=False` is used. Checkpoint files contain NumPy training metadata; newer PyTorch can allow only known scalar/dtype types in a scoped safe unpickler. On older PyTorch, use `--trusted-checkpoint` (or `trusted_pickle=True`) only for a checkpoint you independently trust, or export a tensor-only state dict using a compatible modern environment. Never enable unrestricted pickle for unknown files.

### CUDA environment

Use the original [Vision Mamba installation instructions](https://github.com/hustvl/Vim#envs-for-pretraining), in an isolated Linux/NVIDIA environment. The historical Vim reference uses Python 3.10 and PyTorch 2.1.1+cu118; use matching versions of CUDA, causal-conv1d and the Vim mamba-1p1p1 fork. A modern generic mamba-ssm release is **not** a drop-in substitute.

The new package vendors the released Python Mamba class, so unlike the historical setup there is no need to overwrite your installed `mamba_simple.py`. Its fused operations still come from the original compatible extension. The historical environment recommendation is not a freshly verified CUDA build.

```bash
# After installing the compatible Vim extensions:
python -m pip install -e './modern[dev]'
```

Verify both required fused functions before running inference:

```python
from panmamba import _mamba_simple
assert callable(_mamba_simple.mamba_inner_fn)
assert callable(_mamba_simple.mamba_inner_fn_no_out_proj)
```

## Input and output contract

Use one NPZ per co-registered scene, with finite floating CHW arrays in the same band order and radiometric range as the trained model. No implicit normalization, clipping, resize, quantization or image-mode conversion.

| Key | Shape | Required |
|:--|:--|:--|
| ms | [4,h,w] | Always |
| pan | [1,4h,4w] | Always |
| target | [4,4h,4w] | Supervised training |

For the released 4-band weights, the expected bands are RGB + near-infrared; preserve the exact original dataset channel order and normalization. Use FP32 for reference comparisons. The production model requires compatible CUDA extensions.  Same-sized scenes must be batched together. The output NPZ has an unbatched floating-point `hrms` array with all four bands.

```python
import torch
from panmamba import build_model, predict, load_weights

model = build_model(device="cuda")
load_weights(model, "pan-sharpening/checkpoint/pretrained/Mamba/mambawv2/bestPSNR.pth")
model.eval()
ms = torch.rand(1,4,32,32,device="cuda")
pan = torch.rand(1,1,128,128,device="cuda")
with torch.inference_mode():
    hrms = predict(model, ms, pan)
```

Random inputs above only demonstrate the tensor contract. They are not satellite results.

```bash
panmamba infer --device cuda --checkpoint pan-sharpening/checkpoint/pretrained/Mamba/mambawv2/bestPSNR.pth --input scene.npz --output result.npz
panmamba benchmark --device cuda --height 128 --image-width 128 --batch 1
panmamba train --device cuda --data paired_train --output runs/train
```

The new optional NPZ trainer keeps the active released default objective: L1 with mean reduction divided by 2× the configured batch size, Adam at 5e-4, gradient clipping 4, and CosineAnnealingWarmRestarts(T0=500, Tmult=1, eta_min=5e-8), stepped once per epoch. It is single-device and accepts prepaired data; original data degradation, augmentation, validation metrics, optimizer-resume behavior and exact training run are not reimplemented. The original solver/config are preserved for historical experiments. The modern `--checkpoint` argument initializes weights, not optimizer resume. Benchmarks synchronize CUDA and report synthetic latency and allocated memory, not claimed speedups. No AMP, compiler, alternate kernel or tiling is enabled silently.

## Validation

The CPU suite checks published source hashes, all three tracked checkpoint hashes, exact parameter-name/shape compatibility, strict checkpoint loading, and old/new predictions under a common **test-only** PyTorch SSM reference at 8×12 HR. This is not a fused CUDA benchmark, nor reproduction of the publication's metrics.

Mandatory before merging a release that claims full numerical parity:

```bash
# On a GPU with the original compatible fused extensions:
PANMAMBA_TEST_CUDA=1 PANMAMBA_TEST_SIZE=128 python -m pytest modern/tests/test_release_parity.py -v -s
# For exact comparison against the original hardcoded 1024×1024 reshape:
PANMAMBA_TEST_CUDA=1 PANMAMBA_TEST_SIZE=1024 python -m pytest modern/tests/test_release_parity.py -v -s
```

The smaller size uses only a shape adaptation in the legacy reference; the 1024 test retains the published view dimensions. Both execute the same fused kernels. Use sufficient VRAM for the 1024 test. The CUDA checks fail rather than skip when explicitly requested. Full-data benchmark metrics also require the original test data/protocol, which are not distributed.

[Automated test run](https://github.com/alexhe101/Pan-Mamba/actions/runs/45547608938). Results must be interpreted according to the specific commit and test environment.

## Original code and publication

- [Original model](https://github.com/alexhe101/Pan-Mamba/blob/ec1b9ead9a02b482851be98b7783111880bdd176/pan-sharpening/model/panmamba_baseline_finalversion.py)
- [Original configuration](https://github.com/alexhe101/Pan-Mamba/blob/ec1b9ead9a02b482851be98b7783111880bdd176/pan-sharpening/option.yml)
- [Publication](https://doi.org/10.1016/j.inffus.2024.102779)

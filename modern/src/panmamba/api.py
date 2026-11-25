"""Explicit inference API without registering wrapper parameters."""
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import pickle

import torch
from torch import Tensor, nn

from .network import Net


def build_model(*, device: str | torch.device = "cpu", width: int = 32) -> Net:
    if width < 4 or width % 2:
        raise ValueError("width must be even and at least four")
    if width != 32:
        raise ValueError("The released Pan-Mamba architecture has fixed width 32")
    return Net(num_channels=4, base_filter=width, args=None).to(device)


def validate_inputs(ms: Tensor, pan: Tensor) -> None:
    if ms.ndim != 4 or pan.ndim != 4 or ms.shape[1] != 4 or pan.shape[1] != 1:
        raise ValueError("Expected NCHW MS with four bands and PAN with one")
    if ms.shape[0] != pan.shape[0] or not ms.shape[0]:
        raise ValueError("Batch sizes must be equal and positive")
    if min(ms.shape[-2:]) < 1 or tuple(pan.shape[-2:]) != tuple(v * 4 for v in ms.shape[-2:]):
        raise ValueError("PAN spatial dimensions must equal 4x the MS dimensions")
    if not ms.is_floating_point() or ms.dtype != pan.dtype or ms.device != pan.device:
        raise ValueError("Inputs must share a floating dtype and device")


def predict(model: nn.Module, ms: Tensor, pan: Tensor) -> Tensor:
    """Same released forward, with the unused middle argument hidden."""
    validate_inputs(ms, pan)
    return model(ms, None, pan)


def load_weights(model: nn.Module, path: str | Path, *, trusted_pickle: bool = False) -> nn.Module:
    """Strictly load a bare state dict or released {'net': ...} checkpoint.

    Only an all-keys DataParallel prefix is stripped. No tensors are removed,
    initialized, reshaped or silently ignored. Unsafe legacy metadata requires
    explicit opt-in, only for a checkpoint you independently trust.
    """
    try:
        data = torch.load(path, map_location="cpu", weights_only=not trusted_pickle)
    except pickle.UnpicklingError as exc:
        # Released checkpoints include NumPy scalar training statistics.
        # Permit only these specific metadata types under the restricted
        # unpickler, including NumPy 2's renamed scalar module.
        if trusted_pickle or "numpy" not in str(exc):
            raise
        if not hasattr(torch.serialization, "safe_globals"):
            raise RuntimeError(
                "This PyTorch version cannot safely deserialize legacy NumPy "
                "metadata. Use a newer PyTorch to export the tensor state dict, "
                "or pass trusted_pickle=True only for independently trusted files."
            ) from exc
        import numpy as np
        try:
            from numpy._core.multiarray import scalar
        except ImportError:
            from numpy.core.multiarray import scalar
        types = [np.dtype, scalar]
        if scalar.__module__ != "numpy.core.multiarray":
            types.append((scalar, "numpy.core.multiarray.scalar"))
        if scalar.__module__ != "numpy._core.multiarray":
            types.append((scalar, "numpy._core.multiarray.scalar"))
        types.extend({type(np.dtype(name)) for name in (
            "bool", "float16", "float32", "float64", "int8", "int16",
            "int32", "int64", "uint8", "uint16", "uint32", "uint64"
        )})
        with torch.serialization.safe_globals(types):
            data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, Mapping):
        raise ValueError("Expected a state dictionary or checkpoint mapping")
    for key in ("net", "state_dict", "model", "params"):
        if key in data and isinstance(data[key], Mapping):
            data = data[key]
            break
    if not data or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in data.items()):
        raise ValueError("Checkpoint does not contain a tensor-only model state")
    keys = list(data)
    if all(k.startswith("module.") for k in keys):
        # Preserve any PyTorch version metadata for module-aware loading.
        state = OrderedDict(data)
        if hasattr(data, "_metadata"):
            state._metadata = data._metadata.copy()
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, "module.")
        data = state
    model.load_state_dict(data, strict=True)
    return model

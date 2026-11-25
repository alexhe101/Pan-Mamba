"""One explicitly paired scene per NPZ, with all spectral bands retained."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .api import validate_inputs


def read_sample(path: str | Path, *, require_target: bool = False):
    with np.load(path, allow_pickle=False) as data:
        sample = {}
        for key in ("ms", "pan", "target"):
            if key not in data:
                if key == "target" and not require_target:
                    continue
                raise ValueError(f"{path}: missing {key}")
            x = data[key]
            if x.ndim != 3 or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
                raise ValueError(f"{path}: {key} must be a finite floating CHW array")
            sample[key] = torch.from_numpy(x.astype(np.float32, copy=True))
    validate_inputs(sample["ms"][None], sample["pan"][None])
    if "target" in sample and tuple(sample["target"].shape) != (4, *sample["pan"].shape[-2:]):
        raise ValueError(f"{path}: target shape must be [4,H,W]")
    return sample


class PairedNPZ(Dataset):
    def __init__(self, folder):
        self.files = sorted(Path(folder).glob("*.npz"))
        if not self.files:
            raise ValueError(f"No paired NPZ samples in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return read_sample(self.files[index], require_target=True)

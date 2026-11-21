import json
import random
import torch
import torch.nn.functional as F
import os
import numpy as np
from numpy.lib import format as np_format
import matplotlib
matplotlib.use("Agg") # Must be called before importing pyplot
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

from typing import Callable, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random seed set to: {seed}")

@dataclass
class HologramConfig:
    scene: str
    hologram_file: str
    hologram_dtype: str
    hologram_shape: List[int]
    hologram_kind: str
    wavelength_m: float
    pitch_m: float
    hologram_width_m: float
    object_centre_distance_m: float
    object_depth_range_m: List[float]
    object_width_m: float
    depth_file: str
    depth_dtype: str
    depth_shape: List[int]
    depth_units: str
    array_convention: str
    pixel_position_m: str
    propagation_sign: str

def load_hologram_config(json_path: Path) -> HologramConfig:
    """Read the JSON metadata file and map it into a typed HologramConfig object."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = HologramConfig(**data)
    print(f"[INFO] Loaded configuration for scene: '{config.scene}'")
    return config


def load_hologram_data(
    config: HologramConfig,
    base_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load hologram and ground-truth depth .npy files as PyTorch tensors on the target device."""
    holo_path = base_dir / config.hologram_file
    if not holo_path.exists():
        raise FileNotFoundError(f"Hologram file not found: {holo_path}")

    depth_path = base_dir / config.depth_file
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    holo_np = np.load(holo_path)
    hologram_tensor = torch.from_numpy(holo_np).to(torch.complex64).to(device)

    depth_np = np.load(depth_path)
    depth_tensor = torch.from_numpy(depth_np).to(torch.float32).to(device)

    print(
        f"[INFO] Data loaded on {device}. "
        f"Hologram: {tuple(hologram_tensor.shape)}, Depth GT: {tuple(depth_tensor.shape)}"
    )
    return hologram_tensor, depth_tensor


def save_amplitude_image(
    amplitude_tensor: torch.Tensor,
    filepath: Path,
    cmap: str = "gray",
) -> None:
    """Save a 2D amplitude tensor as a PNG image using origin='lower' convention."""
    amp_np = amplitude_tensor.detach().cpu().numpy()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    vmax = np.percentile(amp_np, 99.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(amp_np, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
    plt.colorbar(label="Amplitude")
    plt.title(f"ASM Reconstruction - {filepath.name}")
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close()


def compute_l1_error(
    pred_depth_idx: np.ndarray | torch.Tensor,
    gt_depth_mm: np.ndarray | torch.Tensor,
    z_planes_m: np.ndarray | torch.Tensor,
    extra_fg_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute foreground-only L1 depth error in millimeters and in plane index units."""

    # Normalize data - np.ndarray on CPU. Convert plane distances from mm to m (multiply with 1000)
    pred_idx = pred_depth_idx.detach().cpu().numpy() if isinstance(pred_depth_idx, torch.Tensor) else np.asarray(pred_depth_idx)
    gt_mm = gt_depth_mm.detach().cpu().numpy() if isinstance(gt_depth_mm, torch.Tensor) else np.asarray(gt_depth_mm)
    z_mm = (z_planes_m.detach().cpu().numpy() if isinstance(z_planes_m, torch.Tensor) else np.asarray(z_planes_m)) * 1000.0

    # Foreground mask - contains True only for object pixels 
    fg_mask = ~np.isnan(gt_mm)
    if extra_fg_mask is not None:
        fg_mask = fg_mask & extra_fg_mask

    # Compute the L1 error
    pred_mm = z_mm[pred_idx.astype(np.int64)]
    l1_mm = float(np.mean(np.abs(gt_mm[fg_mask] - pred_mm[fg_mask])))

    # Compute mean error in number of planes
    dz_mm = (z_mm[-1] - z_mm[0]) / max(len(z_mm) - 1, 1)
    gt_idx = (gt_mm[fg_mask] - z_mm[0]) / dz_mm
    l1_plane_idx = float(np.mean(np.abs(gt_idx - pred_idx[fg_mask])))

    return l1_mm, l1_plane_idx


def apply_zero_padding(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Apply 2D zero-padding of size (patch_size // 2) on the last two spatial dimensions (H, W).
    Works for both 2D planes (H, W) and 3D volumes (N, H, W).
    """
    pad = patch_size // 2
    if pad <= 0:
        return tensor
    # F.pad applies (pad_left, pad_right, pad_top, pad_bottom) to the last two dims
    return F.pad(tensor, (pad, pad, pad, pad), mode="constant", value=0.0)


class PaddedVolume:
    """
    Zero-copy lazy wrapper that applies zero-padding of size (patch_size // 2)
    on-the-fly to each 2D plane when indexed. Avoids duplicating a 16+ GB volume in RAM.
    """

    def __init__(self, volume, patch_size: int):
        self.volume = volume
        self.patch_size = patch_size
        self.pad = patch_size // 2
        N, H, W = volume.shape
        self.shape = (N, H + 2 * self.pad, W + 2 * self.pad)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return apply_zero_padding(self.volume[idx], self.patch_size)
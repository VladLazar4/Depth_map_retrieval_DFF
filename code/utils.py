import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg") # Must be called before importing pyplot
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

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
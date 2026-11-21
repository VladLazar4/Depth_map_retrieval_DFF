import argparse
import time
from pathlib import Path
import torch
import numpy as np

from asm_propagator import AngularSpectrumPropagator
from focus import estimate_depth_map
from evaluate import run_test_focus, run_test_reconstruction
from utils import (
    DATA_DIR,
    RESULTS_DIR,
    load_hologram_config,
    load_hologram_data,
    set_seed,
    PaddedVolume
)


def run_full_pipeline(
    data_dir: Path = DATA_DIR / "rook_hologram_data",
    output_dir: Path = RESULTS_DIR / "full_run",
    num_planes: int = 250, # as mentioned in the paper
    patch_size: int = 13, # 13 is the optimal patch size found in the paper - we should experiment with more values (same as in table 2)
    operator_name: str = "CONT" # choose the desired operator for part 2
) -> None:
    """Run the full DFF holographic depth retrieval pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "metadata.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running full pipeline on device: {device}")

    config = load_hologram_config(config_path)
    H_real, depth_gt = load_hologram_data(config, data_dir, device)

    nx, ny = config.hologram_shape
    propagator = AngularSpectrumPropagator(
        nx=nx,
        ny=ny,
        pixel_pitch=config.pitch_m,
        wavelength=config.wavelength_m,
        device=device,
    )

    z_min, z_max = config.object_depth_range_m
    z_planes = torch.linspace(z_min, z_max, num_planes, device=device)

    # pad = patch_size // 2
    # padded_nx = nx + 2 * pad
    # padded_ny = ny + 2 * pad

    # Pre-allocate reconstruction volume on CPU RAM to avoid GPU OOM at 4096x4096x250 (~16.8 GB) - may need do downsample
    print(f"[INFO] Pre-allocating reconstruction volume ({num_planes}, {nx}, {ny}) in CPU RAM...")
    reconstruction_volume = torch.empty((num_planes, nx, ny), dtype=torch.float32, device="cpu")

    start_time = time.perf_counter()
    for i, z_tensor in enumerate(z_planes):
        # Angular spectrum reconstruction
        amp = propagator.propagate(H = H_real, z = z_tensor.item())
        reconstruction_volume[i] = amp.cpu()
        if (i + 1) % 50 == 0 or i == num_planes - 1:
            print(f"[INFO] Propagated plane {i + 1}/{num_planes}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[INFO] Part 1 (ASM Reconstruction) completed in {time.perf_counter() - start_time:.2f} s.")

    # Save volume for Part 2 or pass directly to Part 2 & Part 3 functions
    volume_path = output_dir / "reconstruction_volume.npy"
    np.save(volume_path, reconstruction_volume.numpy())
    print(f"[INFO] Reconstruction volume saved to: {volume_path}")

    # Part 2 - Focus Measure Operators & Auto-Switch Selection

    # Apply zero-padding
    padded_volume = PaddedVolume(reconstruction_volume, patch_size=patch_size)
    initial_depth_idx = estimate_depth_map(
        reconstruction_volume=padded_volume,
        operator_name=operator_name,
        patch_size=patch_size,
        selection_mode="auto_switch",
        device=device,
    )

    initial_depth_path = output_dir / "initial_depth_idx.npy"
    np.save(initial_depth_path, initial_depth_idx.numpy())
    print(f"[INFO] Initial depth map saved to: {initial_depth_path}")

    # TODO: Part 3 - Voting Agreement & 3D Point Cloud Projection
    # TODO: Part 4 - Quantitative & Qualitative evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holographic Depth-from-Focus (DFF) Pipeline")
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=["full", "test_reconstruction", "test_focus"],
        help="Execution mode: 'full', 'test_reconstruction', or 'test_focus'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_seed()
    args = parse_args()
    if args.mode == "test_reconstruction":
        run_test_reconstruction()
    elif args.mode == "test_focus":
        run_test_focus()
    else:
        run_full_pipeline()
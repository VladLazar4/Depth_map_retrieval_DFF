import time
from pathlib import Path
import cv2
import numpy as np
from numpy.lib import format as np_format
import torch
import os
import matplotlib.pyplot as plt

from utils import compute_l1_error, PaddedVolume
from typing import Callable, Dict, List, Optional, Tuple
from focus import estimate_depth_map

from asm_propagator import AngularSpectrumPropagator
from utils import (
    DATA_DIR,
    RESULTS_DIR,
    load_hologram_config,
    load_hologram_data,
    save_amplitude_image,
)

def _normalize_to_uint8(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Normalize a 2D float amplitude array to uint8 [0, 255] using percentile clipping."""
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def run_test_reconstruction(
    data_dir: Path = DATA_DIR / "rook_hologram_data",
    output_dir: Path = RESULTS_DIR / "tests" / "test_reconstruction",
    num_planes: int = 250,
    patch_size: int = 13,
    save_every: int = 25,
    video_size: int = 1024,
    fps: int = 15,
) -> None:
    """
    Propagate the hologram across all Z planes, generate an annotated MP4 video
    of all reconstructed frames, and save individual PNG snapshots every `save_every` frames.
    Also saves ground-truth in-focus masks (as in Fig. 5 of the paper) for verification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "metadata.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running reconstruction test on device: {device}")

    config = load_hologram_config(config_path)
    H_real, depth_gt = load_hologram_data(config, data_dir, device)

    # Convert ground-truth depth from mm to meters for Fig. 5 in-focus mask verification
    depth_gt_m = (depth_gt / 1000.0).detach().cpu().numpy()

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
    dz = (z_max - z_min) / max(num_planes - 1, 1)

    pad = patch_size // 2
    padded_nx = nx + 2 * pad
    padded_ny = ny + 2 * pad

    video_path = output_dir / "reconstruction_volume.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (video_size, video_size))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    print(f"[INFO] Starting propagation for {num_planes} planes in [{z_min:.5f}, {z_max:.5f}] m...")

    for i, z_tensor in enumerate(z_planes):
        z_val = z_tensor.item()
        amplitude = propagator.propagate(H_real, z_val)

        # Save PNG snapshot every `save_every` frames (plus the very last frame)
        if i % save_every == 0 or i == num_planes - 1:
            img_path = output_dir / f"frame_{i:03d}_z_{z_val:.5f}m.png"
            save_amplitude_image(amplitude, img_path)

            # Save ground-truth in-focus region mask (Fig. 5 verification)
            in_focus_mask = np.abs(depth_gt_m - z_val) <= (dz * 1.5)
            mask_vis = (in_focus_mask.astype(np.uint8) * 255)
            # Flip vertically to match origin='lower' convention from metadata.json
            mask_vis = np.flipud(mask_vis)
            cv2.imwrite(str(output_dir / f"mask_infocus_{i:03d}_z_{z_val:.5f}m.png"), mask_vis)

            print(f"[INFO] Saved snapshot and in-focus mask for frame {i:03d}/{num_planes - 1} (z={z_val:.5f} m)")

        # Prepare frame for video: move to CPU, flip vertically (origin='lower'), normalize, resize
        amp_np = amplitude.detach().cpu().numpy()
        amp_np = np.flipud(amp_np)
        frame_gray = _normalize_to_uint8(amp_np)
        frame_resized = cv2.resize(frame_gray, (video_size, video_size), interpolation=cv2.INTER_AREA)
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

        # Overlay frame index and propagation distance
        label = f"Frame: {i:03d}/{num_planes - 1} | z = {z_val:.5f} m"
        cv2.putText(
            frame_bgr,
            label,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        video_writer.write(frame_bgr)

    video_writer.release()

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[INFO] Peak VRAM allocated: {peak_mem:.2f} GB")

    elapsed = time.perf_counter() - start_time
    print(f"[INFO] Reconstruction test finished in {elapsed:.2f} s.")
    print(f"[INFO] Video saved to: {video_path}")



    
def run_test_focus(
    data_dir: Path = DATA_DIR / "rook_hologram_data",
    output_dir: Path = RESULTS_DIR / "tests" / "test_focus",
    volume_cache_path: Path = RESULTS_DIR / "full_run" / "reconstruction_volume.npy",
    num_planes: int = 250,
    patch_size: int = 13,
    operators: Optional[List[str]] = None,
) -> None:
    """
    Test Part 2 (Focus Measure Operators & Auto-Switch) without Voting Agreement.
    Compares classical argmax vs. Auto-Switch on patch_size=13, prints L1 errors,
    and saves visual comparison panels in results/tests/test_focus/.
    """
    if operators is None:
        operators = ["GLVA", "GLVN", "CONT"]

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "metadata.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[INFO] Running Part 2 focus test on: {device} ({gpu_name})")

    config = load_hologram_config(config_path)
    H_real, depth_gt = load_hologram_data(config, data_dir, device)

    z_min, z_max = config.object_depth_range_m
    z_planes = torch.linspace(z_min, z_max, num_planes, device="cpu")
    z_mm = z_planes.numpy() * 1000.0
    gt_np = depth_gt.detach().cpu().numpy()
    fg_mask = ~np.isnan(gt_np)

    # Load cached reconstruction volume if available, otherwise compute it
    if volume_cache_path.exists():
        print(f"[INFO] Memory-mapping volume from: {volume_cache_path}")
        vol_mmap = np.load(volume_cache_path, mmap_mode="r")
        reconstruction_volume = torch.from_numpy(vol_mmap)
        reconstruction_volume = PaddedVolume(reconstruction_volume, patch_size=patch_size)
    else:
        print(f"[INFO] No cached volume found. Please construct the volumes first and save them to {volume_cache_path}")
        exit(0)

    vmin = float(np.nanmin(gt_np))
    vmax = float(np.nanmax(gt_np))

    print(f"\n=== PART 2 FOCUS TEST (Patch {patch_size}x{patch_size}, No Voting) ===")
    for op in operators:
        pred_argmax = estimate_depth_map(
            reconstruction_volume, op, patch_size=patch_size, selection_mode="argmax", device=device
        )
        l1_mm_arg, l1_idx_arg = compute_l1_error(pred_argmax, gt_np, z_planes)

        pred_switch = estimate_depth_map(
            reconstruction_volume, op, patch_size=patch_size, selection_mode="auto_switch", device=device
        )
        l1_mm_sw, l1_idx_sw = compute_l1_error(pred_switch, gt_np, z_planes)

        print(
            f"[TEST_FOCUS] {op:<4} | "
            f"Argmax L1: {l1_idx_arg:6.2f} idx ({l1_mm_arg:.3f} mm) | "
            f"Auto-Switch L1: {l1_idx_sw:6.2f} idx ({l1_mm_sw:.3f} mm)"
        )

        # Save 4-panel figure: GT | Argmax | Auto-Switch | Auto-Switch Abs Error
        arg_mm = np.where(fg_mask, z_mm[pred_argmax.numpy()], np.nan)
        sw_mm = np.where(fg_mask, z_mm[pred_switch.numpy()], np.nan)
        err_mm = np.where(fg_mask, np.abs(gt_np - sw_mm), np.nan)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Part 2 Test - {op} (Patch {patch_size}x{patch_size}, No Voting)", fontsize=14)

        im0 = axes[0].imshow(gt_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth [mm]")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(arg_mm, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Argmax (L1={l1_idx_arg:.2f} idx)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(sw_mm, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[2].set_title(f"Auto-Switch (L1={l1_idx_sw:.2f} idx)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)

        err_vmax = float(np.nanpercentile(err_mm, 98)) if np.any(fg_mask) else 1.0
        im3 = axes[3].imshow(err_mm, cmap="inferno", origin="lower", vmin=0.0, vmax=err_vmax)
        axes[3].set_title("Auto-Switch Abs Error [mm]")
        fig.colorbar(im3, ax=axes[3], fraction=0.046)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        out_path = output_dir / f"test_focus_{op}_s{patch_size}.png"
        with open(str(out_path), "wb") as f:
            fig.savefig(f, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved Part 2 test figure to: {out_path}")
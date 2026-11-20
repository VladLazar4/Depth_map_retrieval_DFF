import time
from pathlib import Path
import cv2
import numpy as np
import torch

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

    # Exact propagation at object_centre_distance_m (e.g., 120 mm) for reference comparison
    z_center = config.object_centre_distance_m
    amp_center = propagator.propagate(H_real, z_center)
    center_img_path = output_dir / f"center_z_{z_center:.5f}m.png"
    save_amplitude_image(amp_center ** 2, output_dir / f"center_intensity_z_{z_center:.5f}m.png")
    print(f"[INFO] Saved exact center reconstruction at z = {z_center:.5f} m to: {center_img_path}")

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
        amplitude = propagator.propagate(H_real, z_val, pad)

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
import time
from typing import Callable, Dict

import torch
import torch.nn.functional as F

# Global registry for focus measure operators
FOCUS_OPERATORS: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {}


def register_operator(name: str):
    """Decorator to register a new focus measure operator by name."""
    def decorator(fn: Callable[[torch.Tensor, int], torch.Tensor]):
        FOCUS_OPERATORS[name.upper()] = fn
        return fn
    return decorator


# ==============================================================================
# Focus Measure Operators (Extensible via @register_operator)
# Each operator receives a 2D plane (H, W) on the target device and patch_size s,
# returns a 2D focus map of shape (H, W)
# ==============================================================================

@register_operator("GLVA")
def focus_glva(plane: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Gray-Level Variance (GLVA):
    FM(R) = E[R^2] - (E[R])^2 computed over each s x s overlapping patch.
    """
    x = plane.unsqueeze(0).unsqueeze(0)  # (1, 1, H_pad, W_pad) - add 2 dimensions for avg_pool2d functions that expects the format: (batch, nr_ch, H, W)
    mean = F.avg_pool2d(x, kernel_size=patch_size, stride=1)
    mean_sq = F.avg_pool2d(x * x, kernel_size=patch_size, stride=1)
    var = torch.clamp(mean_sq - mean * mean, min=0.0) # In case of small floating point errors
    return var.squeeze(0).squeeze(0) # squeeze - discard redundant dimensions added at the beginning


@register_operator("GLVN")
def focus_glvn(plane: torch.Tensor, patch_size: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Gray-Level Variance (GLVN):
    FM(R) = Var(R) / (Mean(R) + eps) computed over each s x s overlapping patch.
    """
    x = plane.unsqueeze(0).unsqueeze(0) # (1, 1, H_pad, W_pad)
    mean = F.avg_pool2d(x, kernel_size=patch_size, stride=1)
    mean_sq = F.avg_pool2d(x * x, kernel_size=patch_size, stride=1)
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    return (var / (mean + eps)).squeeze(0).squeeze(0) # squeeze - discard redundant dimensions added at the beginning


@register_operator("CONT")
def focus_cont(plane: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Image Contrast (CONT):
    Computes local 8-neighborhood absolute intensity differences C(x, y),
    then aggregates over each s x s overlapping patch.
    """
    # 1. Compute 8-neighborhood contrast map on the plane
    x = plane.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x_pad1 = F.pad(x, (1, 1, 1, 1), mode="replicate").squeeze(0).squeeze(0) # 1-pixel border for 8-neighbors, replicate => copy border values
    H, W = plane.shape

    contrast_map = torch.zeros_like(plane)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            # we consider (dx,dy) neghhbors for all pixels in the plane 
            # [linie_start : linie_final, col_start : col_final]
            neighbor = x_pad1[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
            contrast_map += torch.abs(plane - neighbor)

    # avg_pool2d is equivalent to sum over s x s up to a constant factor (s^2)
    fm = F.avg_pool2d(contrast_map.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=1)
    return fm.squeeze(0).squeeze(0)

# ==============================================================================
# ESTIMATE DEPTH MAP
# ==============================================================================


def estimate_depth_map(
    reconstruction_volume: torch.Tensor,
    operator_name: str,
    patch_size: int = 13,
    selection_mode: str = "auto_switch",
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Estimate the initial pixel-wise depth map (plane indices) from the 3D amplitude volume.

    Args:
        reconstruction_volume: 3D tensor (N, H, W) of reconstructed amplitudes (CPU or GPU).
        operator_name: Focus measure operator name (e.g., 'GLVA', 'GLVN', 'CONT').
        patch_size: Sliding window size s (odd integer, e.g., 5, 9, 13, 17, 21).
        selection_mode: 'argmax' (classical DFF) or 'auto_switch' (Madali et al., 2023).
        device: Target compute device (defaults to CUDA if available).

    Returns:
        depth_idx: 2D int64 tensor (H, W) with optimal plane index in [0, N-1] on CPU.
    """
    op_key = operator_name.upper()
    if op_key not in FOCUS_OPERATORS:
        available = ", ".join(sorted(FOCUS_OPERATORS.keys()))
        raise ValueError(f"Unknown focus operator '{operator_name}'. Available: {available}")

    if selection_mode not in ("argmax", "auto_switch"):
        raise ValueError(f"Invalid selection_mode '{selection_mode}'. Use 'argmax' or 'auto_switch'.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fm_fn = FOCUS_OPERATORS[op_key]
    num_planes, H_pad, W_pad = reconstruction_volume.shape
    H_out = H_pad - patch_size + 1
    W_out = W_pad - patch_size + 1
    start_time = time.perf_counter()

    print(
        f"[INFO] Part 2 | Operator={op_key} | Patch={patch_size}x{patch_size} | "
        f"Selection={selection_mode} | Device={device}"
    )

    with torch.no_grad():
        if selection_mode == "argmax":
            # Single pass: find argmax_i FM(R_{i,m,n})
            best_score = torch.full((H_out, W_out), -float("inf"), dtype=torch.float32, device=device) # create a tensor of size (H_out, W_out) filled with -inf values
            best_idx = torch.zeros((H_out, W_out), dtype=torch.int64, device=device) # create a tensor of size (H_out, W_out) filled with 0 values

            for i in range(num_planes):
                plane_i = reconstruction_volume[i].to(device, non_blocking=True)
                fm_i = fm_fn(plane_i, patch_size)

                mask = fm_i > best_score
                best_score[mask] = fm_i[mask]
                best_idx[mask] = i

        else: # Auto-Switch

            # Compute mean focus value across all N planes: d_bar = (1/N) * sum_i FM_i
            fm_sum = torch.zeros((H_out, W_out), dtype=torch.float32, device=device)
            for i in range(num_planes):
                plane_i = reconstruction_volume[i].to(device, non_blocking=True)
                fm_sum += fm_fn(plane_i, patch_size)
            fm_mean = fm_sum / float(num_planes) # d_bar

            # Find argmax_i |d_bar - FM(R_{i,m,n})|
            max_dev = torch.full((H_out, W_out), -float("inf"), dtype=torch.float32, device=device)
            best_idx = torch.zeros((H_out, W_out), dtype=torch.int64, device=device)

            for i in range(num_planes):
                plane_i = reconstruction_volume[i].to(device, non_blocking=True)
                fm_i = fm_fn(plane_i, patch_size)
                dev_i = torch.abs(fm_mean - fm_i)

                mask = dev_i > max_dev
                max_dev[mask] = dev_i[mask]
                best_idx[mask] = i

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    print(f"[INFO] Part 2 completed in {elapsed:.2f} s.")
    return best_idx.cpu()
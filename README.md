# Automatic Depth Map Retrieval from Digital Holograms Using a Depth-from-Focus Approach

Reproduction project for the **Computer Vision** course, implementing the Depth-from-Focus (DFF) pipeline for computer-generated holograms (CGHs) with Angular Spectrum Method (ASM) reconstruction, Auto-Switch optimal plane selection, and Voting Agreement.

---

## Paper Citation

> N. Madali, A. Gilles, P. Gioia, and L. Morin, *"Automatic depth map retrieval from digital holograms using a depth-from-focus approach,"* **Applied Optics**, Vol. 62, No. 10, pp. D77–D89, April 2023.  
> DOI: [https://doi.org/10.1364/AO.478634](https://doi.org/10.1364/AO.478634)

A copy of the original paper is included in the project root as `paper.pdf`.

---

## Team

* **Vlad Lazar**
* **Kawtar Bidoukhach**
* **Khadija Gaizi**

---

## Repository Structure

```text
Depth_map_retrieval_DFF/
├── code/
│   ├── asm_propagator.py     # Angular Spectrum Method (ASM) propagation & zero-padding
│   ├── evaluate.py           # Reconstruction verification (MP4 video + snapshots + GT masks)
│   ├── main.py               # Main entry point for full pipeline and tests
│   └── utils.py              # Relative path helpers, seed setup, data loading & saving
├── data/
│   ├── README-data.txt       # Dataset source, structure, and download instructions
│   └── rook_hologram_data/   # Input hologram (.npy), depth GT (.npy), and metadata.json
├── results/
│   ├── figures/              # Qualitative outputs (depth maps, all-in-focus images, point clouds)
│   ├── tests/                # Reconstruction test videos and snapshots
│   └── metrics.csv           # Quantitative evaluation results (L1 error)
├── algorithm.txt             # High-level pseudocode of the core DFF algorithm
├── paper.pdf                 # Original research paper
├── report.pdf                # Final project report
├── requirements.txt          # Python package dependencies
├── run.sh                    # Single-command execution wrapper script
└── README.md                 # Project documentation
```

---

## Dataset

The experiments are evaluated on a complex-valued digital hologram of a chess rook scene (`4096 x 4096`, `complex64`, $\lambda = 520\text{ nm}$, pixel pitch $= 3.72\ \mu\text{m}$, depth range $[0.11496, 0.12406]\text{ m}$) with its corresponding ground-truth depth map (`rook_depth_mm.npy`).

Due to file size constraints, raw `.npy` files may not be included directly in the submission archive. See `data/README-data.txt` for download links and setup instructions.

---

## Environment & Requirements

* **Python:** 3.10 – 3.12 (64-bit)
* **Dependencies:** Listed in `requirements.txt` (`torch`, `numpy`, `opencv-python`, `matplotlib`)
* **Reproducibility:** Random seeds for Python, NumPy, and PyTorch are fixed to `42` via `set_seed(42)` in `code/utils.py`.
* **Paths:** All paths are strictly relative to the project root (no hardcoded absolute paths).

---

## How to Run

The wrapper script `run.sh` automatically creates a local virtual environment (`.venv`), installs the required packages from `requirements.txt` (with CUDA support if an NVIDIA GPU is detected), and executes the code without requiring any manual edits.

### 1. Run the Full DFF Pipeline (Default)
```bash
bash run.sh
```
This runs the complete pipeline: ASM numerical reconstruction across $N=250$ depth planes with zero-padding, focus measurement with Auto-Switch selection, Voting Agreement, and metric computation (saving outputs to `results/`).

### 2. Run the ASM Reconstruction Test (Part 1 Verification)
```bash
bash run.sh test_reconstruction
```
This propagates the hologram across all 250 planes and outputs:
* An annotated MP4 video (`results/tests/test_reconstruction/reconstruction_volume.mp4`) showing focus progression across $z$.
* PNG amplitude snapshots and corresponding ground-truth in-focus masks every 25 frames.

---

## Hardware & Runtime

* **Hardware Used:** `NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)`
* **Approximate Runtime:**
  * Part 1 (ASM Reconstruction, 250 planes @ `4096 x 4096`): `214.2 seconds`, with the peak allocated VRAM of `1.20GB`
  * Full Pipeline (Parts 1 + 2 + 3): `[TODO: X minutes]`

---

## Expected Outputs & Verification Metrics

Running `bash run.sh` produces the reconstructed depth map, all-in-focus image, and `results/metrics.csv`.

### Reference Metrics from the Original Paper (L1 Error, $13 \times 13$ patch with Voting & Auto-Switch)
| Focus Operator | L1 Error (Paper) |
| :--- | :---: |
| **CONT** (Image Contrast) | `1.54` |
| **GLVA** (Gray-Level Variance) | `1.91` |
| **GLVN** (Normalized Gray-Level Variance) | `2.99` |

### Expected Results on Our Dataset (`rook_hologram.npy`)
| Focus Operator | Patch Size | Voting | Expected L1 Error |
| :--- | :---: | :---: | :---: |
| **CONT** | `13 x 13` | Yes | `[TODO: fill after Part 2 & 3]` |
| **GLVA** | `13 x 13` | Yes | `[TODO: fill after Part 2 & 3]` |
| **GLVN** | `13 x 13` | Yes | `[TODO: fill after Part 2 & 3]` |
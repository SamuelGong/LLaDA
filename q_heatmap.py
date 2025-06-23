"""q_heatmap.py – interactive & auto‑save

Inspect logged Query vectors (saved by `q_logging_test.py`) and produce a heatmap
of pair‑wise cosine similarities across diffusion steps.

Features
────────
1. **Automatic discovery** of recorded steps, layers and sequence length.
2. Prompts you to choose a *layer index* and *token index* interactively.
3. **Auto‑saves** the figure to:
       <root>/plots/heatmap_L{layer}_T{token}.png
   creating the `plots/` directory if necessary. No GUI pop‑up.

Usage
─────
    python q_heatmap.py                # assumes q_cache as root
    python q_heatmap.py --root logs/   # custom directory
"""

import argparse
import glob
import os
import re
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless backend – no window
import matplotlib.pyplot as plt
import numpy as np
import torch

# ──────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────

def _list_steps(root: str) -> List[int]:
    return sorted(
        int(m.group(1))
        for p in glob.glob(os.path.join(root, "step_*"))
        if (m := re.search(r"step_(\d+)", p))
    )


def _list_layers(root: str, step: int) -> List[int]:
    return sorted(
        int(m.group(1))
        for p in glob.glob(os.path.join(root, f"step_{step:04d}", "layer_*"))
        if (m := re.search(r"layer_(\d+)", p))
    )


def _load_vecs(root: str, layer: int, token: int, steps: List[int]) -> torch.Tensor:
    vecs = []
    for s in steps:
        path = os.path.join(root, f"step_{s:04d}", f"layer_{layer:03d}", "q.bin")
        q = torch.load(path)
        vecs.append(q[0, token])  # batch 0
    return torch.stack(vecs)  # (T, d)


def _cosine(mat: torch.Tensor) -> np.ndarray:
    """Return (T, T) cosine‑similarity matrix in float32 to avoid bf16 issues."""
    mat = mat.to(torch.float32)  # ensure dtype supported by matmul & numpy
    mat = torch.nn.functional.normalize(mat, dim=-1)
    return (mat @ mat.T).cpu().numpy()

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="q_cache", help="Dir with step_*/layer_*/q.bin")
    args = parser.parse_args()
    root = args.root.rstrip("/")

    if not os.path.isdir(root):
        raise SystemExit(f"Directory not found: {root}")

    steps = _list_steps(root)
    if not steps:
        raise SystemExit("No step_* directories detected.")
    first_step = steps[0]

    layers = _list_layers(root, first_step)
    if not layers:
        raise SystemExit("No layer_* directories detected in first step.")

    # Infer seq length
    sample_q = torch.load(os.path.join(root, f"step_{first_step:04d}", f"layer_{layers[0]:03d}", "q.bin"))
    seq_len = sample_q.shape[1]

    print("Detected:")
    print(f"  Steps: {len(steps)} (0–{len(steps)-1})")
    print(f"  Layers: 0–{layers[-1]}")
    print(f"  Token positions: 0–{seq_len-1}\n")

    # Prompt user
    while True:
        try:
            layer = int(input("Select layer index: "))
            token = int(input("Select token index: "))
        except ValueError:
            print("Please enter valid integers.")
            continue
        if layer not in layers:
            print("Layer out of range.")
            continue
        if not (0 <= token < seq_len):
            print("Token out of range.")
            continue
        break

    print(f"Loading vectors for layer {layer}, token {token} …")
    vecs = _load_vecs(root, layer, token, steps)
    sim = _cosine(vecs)

    # Plot (headless)
    plt.figure(figsize=(6, 5))
    plt.imshow(sim, origin="lower", aspect="auto")
    plt.colorbar(label="cosine similarity")
    plt.xlabel("step j")
    plt.ylabel("step i")
    plt.title(f"Layer {layer}, token {token}")

    out_dir = os.path.join(root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_L{layer}_T{token}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")

    print("Saved →", out_path)


if __name__ == "__main__":
    main()

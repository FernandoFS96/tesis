"""Data loading utilities for magnetic-trajectory ANP experiments (batched).

This version **pads and stacks** multiple trajectories inside the
``episodic_collate`` so that ``batch_size > 1`` in the ``DataLoader`` works
properly – no episodes are dropped.  It also returns **Boolean masks** that
flag padded positions, making it easy to extend the ANP to ignore padding
(by passing `key_padding_mask` to each ``nn.MultiheadAttention`` layer).

"""
from __future__ import annotations

import glob
import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # ANP root
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 7)]
Y_COLS = ["object_x", "object_y"]
META_DIM = 3  # depth, length, width

# -----------------------------------------------------------------------------
#  Helper: parse metadata from folder name
# -----------------------------------------------------------------------------

def parse_meta(folder_name: str) -> Tuple[float, float, float]:
    """Extract (depth, length, width) from folder names like '7-5m-64T_2mx1m'."""
    # Depth
    depth_match = re.search(r"([0-9]+(?:-[0-9])?)m", folder_name)
    depth = float(depth_match.group(1).replace("-", ".")) if depth_match else 0.0

    # Size (length × width)
    size_match = re.search(r"_([0-9]+)mx([0-9]+)m", folder_name)
    length = float(size_match.group(1)) if size_match else 0.0
    width = float(size_match.group(2)) if size_match else 0.0

    return depth, length, width


# -----------------------------------------------------------------------------
#  Dataset class
# -----------------------------------------------------------------------------
class MagneticTrajectoryDataset(Dataset):
    """Loads one trajectory (100 points) per sample."""

    def __init__(
        self,
        root: str | Path,
        *,
        use_meta: bool = False,
        verbose: bool = True,
        cache: bool = True,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.use_meta = use_meta
        self.verbose = verbose

        #for csv_path in sorted(glob.glob(pattern)):
        pattern1 = os.path.join(str(root), "*", "dataset.csv")
        pattern2 = os.path.join(str(root), "dataset.csv")
        pattern3 = os.path.join(str(root), "*", "dataset*.csv")

        files = glob.glob(pattern1) or glob.glob(pattern2) or glob.glob(pattern3)
        for csv_path in sorted(files):
            folder = os.path.basename(os.path.dirname(csv_path))
            depth, length, width = parse_meta(folder)
            meta_tensor = torch.tensor([depth, length, width], dtype=torch.float32)

            #df = pd.read_csv(csv_path)

            df = pd.read_csv(csv_path)

            # ------------------------------------------------------------------
            #  Asegurarse de que siempre existan las 6 columnas sensor_*.
            #  Si el CSV carece de alguna, la creamos y la rellenamos con 0.0
            # ------------------------------------------------------------------
            missing = [c for c in SENSOR_COLS if c not in df.columns]
            if missing:
                if self.verbose:
                    print(f"[WARN] {csv_path} no contiene {missing} → rellenando con 0")
                for col in missing:
                    df[col] = 0.0
            # Garantizar el orden fijo de las columnas
            df = df.sort_index(axis=1)           # o bien df = df[SENSOR_COLS + Y_COLS + ...]
            

            for _, traj_df in df.groupby("traj_id"):
                sensors = torch.as_tensor(traj_df[SENSOR_COLS].values, dtype=torch.float32)
                coords = torch.as_tensor(traj_df[Y_COLS].values, dtype=torch.float32)

                if self.use_meta:
                    meta_rep = meta_tensor.repeat(sensors.size(0), 1)
                    sensors = torch.cat([sensors, meta_rep], dim=-1)

                if cache:
                    self.samples.append((sensors, coords))
                else:
                    self.samples.append((sensors.numpy(), coords.numpy()))
            if self.verbose:
                print(f"Loaded from {csv_path}: {len(df['traj_id'].unique())} trajectories")
        if not self.samples:
            raise RuntimeError(f"No trajectories found under {root}")
        self.x_dim = self.samples[0][0].shape[1]
        if self.verbose:
            print(f"[Dataset] {len(self.samples)} trayectorias | x_dim = {self.x_dim}")

    # ------------------------------ PyTorch API ------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sensors, coords = self.samples[idx]
        if not isinstance(sensors, torch.Tensor):
            sensors = torch.as_tensor(sensors, dtype=torch.float32)
            coords = torch.as_tensor(coords, dtype=torch.float32)
        return sensors, coords


# -----------------------------------------------------------------------------
#  Collate (batched & padded)
# -----------------------------------------------------------------------------

def _pad_sequence(seq: torch.Tensor, max_len: int) -> torch.Tensor:
    """Right‑pad along the N (trajectory‑length) dimension with zeros."""
    pad_len = max_len - seq.size(0)
    if pad_len == 0:
        return seq
    return F.pad(seq, (0, 0, 0, pad_len))  # (left,right) for last dim, then (top,bottom)


def episodic_collate(batch, *, min_ctx: int = 15, max_ctx: int = 55,
                     ctx_mode: str = "random",          # "random" | "sequential"
                     fixed_ctx_size: int | None = None    # if set, always use this many context points
                     ):
    """Return padded tensors so `batch_size > 1` is supported without drops.

    Outputs
    -------
    x_c, y_c : torch.Tensor
        ``(B, L_ctx, x_dim)`` and ``(B, L_ctx, 2)``
    x_t, y_t : torch.Tensor
        ``(B, L_tgt, x_dim)`` and ``(B, L_tgt, 2)``
    ctx_mask, tgt_mask : torch.BoolTensor
        ``True`` at padding positions (shape ``(B, L_*)``).  Can be ignored
        if you do not mask attention heads yet.
    """
    xs_c, ys_c, xs_t, ys_t = [], [], [], []
    ctx_lens, tgt_lens = [], []

    for sensors, coords in batch:
        N = sensors.size(0)

        #idx = torch.randperm(N)
        #n_ctx = random.randint(min_ctx, min(max_ctx, N - 1))
        #c_idx, t_idx = idx[:n_ctx], idx[n_ctx:]
        if ctx_mode == "random":                    # ← existing behaviour
            idx = torch.randperm(N)
            if fixed_ctx_size is not None:
                n_ctx = min(fixed_ctx_size, N - 1)
            else:
                n_ctx = random.randint(min_ctx, min(max_ctx, N - 1))
            c_idx, t_idx = idx[:n_ctx], idx[n_ctx:]

        elif ctx_mode == "sequential":              # ← new extrapolation mode
            if fixed_ctx_size is not None:
                n_ctx = min(fixed_ctx_size, N - 1)
            else:
                choices = [10, 20, 30, 40, 50, 60]
                n_ctx = random.choice([c for c in choices if c < N])
            c_idx = torch.arange(n_ctx)            # first n_ctx indices
            t_idx = torch.arange(n_ctx, N)         # the rest

        else:
            raise ValueError("ctx_mode must be 'random' or 'sequential'")

        x_c, y_c = sensors[c_idx], coords[c_idx]
        x_t, y_t = sensors[t_idx], coords[t_idx]

        xs_c.append(x_c)
        ys_c.append(y_c)
        xs_t.append(x_t)
        ys_t.append(y_t)
        ctx_lens.append(x_c.size(0))
        tgt_lens.append(x_t.size(0))

    max_ctx_len = max(ctx_lens)
    max_tgt_len = max(tgt_lens)

    pad_ctx = lambda seq: _pad_sequence(seq, max_ctx_len)
    pad_tgt = lambda seq: _pad_sequence(seq, max_tgt_len)

    x_c = torch.stack([pad_ctx(x) for x in xs_c])  # (B, L_ctx, x_dim)
    y_c = torch.stack([pad_ctx(y) for y in ys_c])  # (B, L_ctx, 2)
    x_t = torch.stack([pad_tgt(x) for x in xs_t])  # (B, L_tgt, x_dim)
    y_t = torch.stack([pad_tgt(y) for y in ys_t])  # (B, L_tgt, 2)

    # Build masks: True means PAD
    ctx_mask = torch.arange(max_ctx_len).unsqueeze(0) >= torch.tensor(ctx_lens).unsqueeze(1)
    tgt_mask = torch.arange(max_tgt_len).unsqueeze(0) >= torch.tensor(tgt_lens).unsqueeze(1)

    return x_c, y_c, x_t, y_t, ctx_mask, tgt_mask

# -----------------------------------------------------------------------------
#  Convenience builder
# -----------------------------------------------------------------------------

def build_dataloader(
    root: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> Tuple[DataLoader, int]:
    ds = MagneticTrajectoryDataset(root, **dataset_kwargs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: episodic_collate(b),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, ds.x_dim


# -----------------------------------------------------------------------------
#  Quick CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check for batched data loader")
    parser.add_argument("root", help="Root folder containing the 16 datasets")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--meta", action="store_true")
    args = parser.parse_args()

    loader, x_dim = build_dataloader(args.root, batch_size=args.batch, use_meta=args.meta)
    print(f"x_dim = {x_dim}")
    for step, (x_c, y_c, x_t, y_t, cm, tm) in enumerate(loader):
        print(f"Batch {step} | ctx {x_c.shape} | tgt {x_t.shape}")
        if step == 2:
            break

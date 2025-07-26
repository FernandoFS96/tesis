import os, re, glob
from pathlib import Path
import random

import torch
# ---------- ANP evaluation helper ----------
@torch.inference_mode()
def mae_anp_fulltraj(model, sensors, coords, device: str,
                     ctx: int = 9) -> float:
    """
    Devuelve el MAE del ANP sobre **todos** los puntos de la trayectoria.
    El contexto (n_ctx) sigue sorteándose en [min_ctx, max_ctx].
    """
    N = sensors.size(0)
    idx = torch.randperm(N)
    n_ctx = ctx #random.randint(min_ctx, min(max_ctx, N - 1))
    c_idx = idx[:n_ctx]

    x_c = sensors[c_idx].unsqueeze(0).to(device)      # (1, n_ctx, x_dim)
    y_c = coords[c_idx].unsqueeze(0).to(device)       # (1, n_ctx, 2)
    x_t = sensors.unsqueeze(0).to(device)             # (1, N,     x_dim)
    y_t = coords.unsqueeze(0).to(device)              # (1, N,     2)

    ctx_mask = torch.zeros(1, n_ctx, dtype=torch.bool, device=device)
    tgt_mask = torch.zeros(1, N,     dtype=torch.bool, device=device)

    dist, _ = model.forward(x_c, y_c, x_t,
                            ctx_mask=ctx_mask, tgt_mask=tgt_mask)
    mae = (dist.mean - y_t).abs().mean().item()       # promedio sobre los N puntos
    return mae
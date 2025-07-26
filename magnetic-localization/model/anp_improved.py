"""anp_combined.py  (mask-aware)
===============================
Attentive Neural Process que acepta máscaras de padding para lotes con
varias trayectorias de longitudes distintas.

Cambios clave
-------------
* Todos los bloques de cross-attention reciben `key_mask`.
* `forward` y `log_likelihood` aceptan `ctx_mask` y `tgt_mask`.
* El log-likelihood ignora posiciones de padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.sensor_dropout import OrderedSensorMask
# --------------------------------------------------------------------------
#  Componentes auxiliares
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # (B, N, D)
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention con dimensiones de query y key distintas."""
    def __init__(self, q_dim: int, ctx_x_dim: int, ctx_y_dim: int,
                 hidden: int, n_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden)
        self.k_proj = nn.Linear(ctx_x_dim, hidden)
        self.v_proj = nn.Linear(ctx_x_dim + ctx_y_dim, hidden)
        self.attn   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden)
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x_q, x_k, y_k, key_mask=None):
        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(torch.cat([x_k, y_k], -1))
        h, _ = self.attn(q, k, v, key_padding_mask=key_mask)
        h = self.ln1(q + h)
        return self.ln2(h + self.ffn(h))


#class LatentEncoder(nn.Module):
#    def __init__(self, x_dim: int, y_dim: int, hidden: int, latent: int):
#        super().__init__()
#        self.mlp = MLP(x_dim + y_dim, hidden, hidden, 3)
#        self.mu      = nn.Linear(hidden, latent)
#        self.log_var = nn.Linear(hidden, latent)
#
#    def forward(self, x, y):
#        h = self.mlp(torch.cat([x, y], -1)).mean(1)
#        return self.mu(h), self.log_var(h)

# --------------------------------------------------------------------------
#  Latent encoder (padding-aware)
# --------------------------------------------------------------------------
class LatentEncoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden: int, latent: int):
        super().__init__()
        self.mlp  = MLP(x_dim + y_dim, hidden, hidden, 3)
        self.mu   = nn.Linear(hidden, latent)
        self.logv = nn.Linear(hidden, latent)

    @staticmethod
    def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h    : (B, L, H)   hidden features
        mask : (B, L)      True = real data, False = padding
        returns (B, H)     mean over TRUE positions only
        """
        w = mask.float()
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)     # normalise
        return (h * w.unsqueeze(-1)).sum(1)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        h = self.mlp(torch.cat([x, y], dim=-1))            # (B, L, H)
        h = self.masked_mean(h, mask)                      # (B, H)
        return self.mu(h), self.logv(h)
    

# --------------------------------------------------------------------------
#  ANP principal
# --------------------------------------------------------------------------
@dataclass
class ANPConfig:
    x_dim: int
    y_dim: int
    hidden_dim: int = 128
    latent_dim: int = 128
    n_heads: int = 8
    enc_layers: int = 2
    dec_layers: int = 3
    sensor_mask: bool = False  # o False para run sin apagado


class ANP(nn.Module):
    def __init__(self, cfg: ANPConfig):
        super().__init__()
        self.cfg = cfg
        self.mask = OrderedSensorMask(order=[0, 2, 4], min_keep=3, enabled=cfg.sensor_mask)

        # Bloques de atención
        self.cross = nn.ModuleList()
        # primer bloque: query de 13 → hidden
        self.cross.append(
            CrossAttention(cfg.x_dim, cfg.x_dim, cfg.y_dim,
                           cfg.hidden_dim, cfg.n_heads)
        )
        # bloques restantes: query ya en 128
        for _ in range(1, cfg.enc_layers):
            self.cross.append(
                CrossAttention(cfg.hidden_dim, cfg.x_dim, cfg.y_dim,
                               cfg.hidden_dim, cfg.n_heads)
            )
        self.latent_enc = LatentEncoder(cfg.x_dim, cfg.y_dim,
                                        cfg.hidden_dim, cfg.latent_dim)
        dec_in = cfg.hidden_dim + cfg.latent_dim + cfg.x_dim
        self.decoder = MLP(dec_in, cfg.hidden_dim, cfg.y_dim * 2, cfg.dec_layers)

    # ---------------- utils ----------------
    @staticmethod
    def _pos_std(log_std):
        return 1e-3 + F.softplus(log_std)

    def _r(self, x_c, y_c, x_t, ctx_mask=None):
        h = x_t
        for blk in self.cross:
            h = blk(h, x_c, y_c, key_mask=ctx_mask)
        return h

    #def _latent(self, x, y):
    #    mu, log_v = self.latent_enc(x, y)
    #    return mu, self._pos_std(log_v)
    def _latent(self, x, y, mask):
        mu, log_v = self.latent_enc(x, y, mask)
        return mu, self._pos_std(log_v)

    # ---------------- forward --------------
    def forward(
        self, x_c, y_c, x_t, y_t=None,
        *, ctx_mask=None, tgt_mask=None
    ):  
        # ---------- MISMA MÁSCARA PARA AMBOS -----------------
        x_c, sens_mask = self.mask(x_c, return_mask=True)   # genera & guarda
        x_t = self.mask(x_t, mask=sens_mask)                # reutiliza sin azar

        # ---------- resto del pipeline ----------------------
        r = self._r(x_c, y_c, x_t, ctx_mask)

        mu_p, std_p = self._latent(x_c, y_c, ctx_mask)
        prior = Normal(mu_p, std_p)

        if y_t is not None:  # entrenamiento
            full_mask   = torch.cat([ctx_mask, tgt_mask], dim=1)
            mu_q, std_q = self._latent(torch.cat([x_c, x_t], 1),
                                       torch.cat([y_c, y_t], 1),
                                       full_mask)
            post = Normal(mu_q, std_q)
            z = post.rsample()
            kl = kl_divergence(post, prior).mean()
        else:                # inferencia
            z  = prior.rsample()
            kl = torch.tensor(0., device=x_c.device)

        z_exp = z.unsqueeze(1).expand(-1, x_t.size(1), -1)
        dec_in = torch.cat([r, z_exp, x_t], -1)
        mu, log_s = self.decoder(dec_in).chunk(2, -1)
        dist = Normal(mu, self._pos_std(log_s))
        return dist, kl

    # --------------- loss API --------------
    def log_likelihood(
        self, x_c, y_c, x_t, y_t, *, ctx_mask=None, tgt_mask=None
    ):
        dist, kl = self.forward(x_c, y_c, x_t, y_t,
                                ctx_mask=ctx_mask, tgt_mask=tgt_mask)
        log_p = dist.log_prob(y_t).sum(-1)          # (B, L_t)
        if tgt_mask is not None:
            log_p = log_p.masked_fill(tgt_mask, 0.0)
            denom = (~tgt_mask).sum() + 1e-6
            log_p = log_p.sum() / denom
        else:
            log_p = log_p.mean()
        return -(log_p - kl)

    # ------------- predicción -------------
    @torch.inference_mode()
    def predict(self, x_c, y_c, x_t, *, ctx_mask=None, n_samples=1):
        mus, sigmas = [], []
        for _ in range(n_samples):
            dist, _ = self.forward(x_c, y_c, x_t, ctx_mask=ctx_mask)
            mus.append(dist.mean)
            sigmas.append(dist.stddev)
        return torch.stack(mus).mean(0), torch.stack(sigmas).mean(0)


# Pequeño test rápido
if __name__ == "__main__":
    cfg = ANPConfig(x_dim=13, y_dim=2)
    model = ANP(cfg)
    x_c = torch.randn(4, 30, 13)
    y_c = torch.randn(4, 30, 2)
    x_t = torch.randn(4, 60, 13)
    y_t = torch.randn(4, 60, 2)
    ctx_m = torch.zeros(4, 30, dtype=torch.bool)
    tgt_m = torch.zeros(4, 60, dtype=torch.bool)
    loss = model.log_likelihood(x_c, y_c, x_t, y_t,
                                ctx_mask=ctx_m, tgt_mask=tgt_m)
    print("loss", loss.item())

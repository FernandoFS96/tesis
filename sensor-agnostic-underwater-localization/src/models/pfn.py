"""Prior-Data Fitted Network (Mueller et al., 2022; Hollmann et al., 2023).

Same interface as the models in anp.py (forward signature, the 5-tuple
(mean, var, loss, kl, nll) and the optional spatial front end), so it plugs into
the same trainer and evaluation scripts.

Context and target points form one sequence processed by a stack of pre-LN
transformer blocks. Following TabPFN, context tokens attend to each other and
target tokens attend to the context only, never to each other or to
themselves, so each target gets an independent prediction. There is no
positional encoding on the point axis, so the context is treated as a set.
The model has no latent variable; the loss is the Gaussian NLL.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .anp import Linear, build_spatial_encoder, gaussian_nll, reduce_nll, DEFAULT_VAR_FLOOR


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (attention + feed-forward).

    key_mask (B, L) marks which sequence positions may be used as keys. It acts
    on the key axis only, so masking the target positions hides them from every
    query, including themselves.
    """

    def __init__(self, d_model, n_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model {d_model} not divisible by n_heads {n_heads}"
        self.h = n_heads
        self.dk = d_model // n_heads
        self.p_drop = dropout
        self.q = Linear(d_model, d_model, bias=False)
        self.k = Linear(d_model, d_model, bias=False)
        self.v = Linear(d_model, d_model, bias=False)
        self.o = Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            Linear(d_model, ffn_mult * d_model, w_init='relu'),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, key_mask=None):
        B, L, D = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(h).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(h).view(B, L, self.h, self.dk).transpose(1, 2)
            # (B, 1, 1, L): the same keys for every head and query.
        am = key_mask[:, None, None, :] if key_mask is not None else None
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am,
            dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.o(a))
        x = x + self.ffn(self.ln2(x))
        return x


class CrossBlock(nn.Module):
    """Cross-attention block with separate query and key-value streams, plus a
    feed-forward sublayer. Used by readout="cross": queries and keys come from
    the acoustics only, and the values carry the context encoding."""

    def __init__(self, d_model, n_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.p_drop = dropout
        self.q = Linear(d_model, d_model, bias=False)
        self.k = Linear(d_model, d_model, bias=False)
        self.v = Linear(d_model, d_model, bias=False)
        self.o = Linear(d_model, d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            Linear(d_model, ffn_mult * d_model, w_init='relu'), nn.GELU(),
            nn.Dropout(dropout), Linear(ffn_mult * d_model, d_model))

    def forward(self, q_in, k_in, v_in, key_mask=None):
        B, Lq, D = q_in.shape
        Lk = k_in.size(1)
        q = self.q(self.ln_q(q_in)).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k = self.k(self.ln_kv(k_in)).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v(self.ln_kv(v_in)).view(B, Lk, self.h, self.dk).transpose(1, 2)
        am = key_mask[:, None, None, :] if key_mask is not None else None
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am, dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, Lq, D)
        x = q_in + self.drop(self.o(a))
        return x + self.ffn(self.ln2(x))


class PFNModel(nn.Module):
    """Transformer neural process with two readouts.

    readout="joint" (default): one stack over [context, target], with targets
        attending to the context only (TabPFN).
    readout="cross": the stack encodes the context, and the targets read it out
        through cross-attention whose queries and keys come from the acoustics
        alone.
    """

    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1,
                 spatial_cfg=None, n_layers=6, n_heads=8, ffn_mult=4,
                 readout="joint", n_cross_layers=2):
        super().__init__()
        self.spatial_encoder, enc_dim = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.output_dim = output_dim
        self.x_proj = Linear(enc_dim, num_hidden, w_init='relu')
        self.y_proj = Linear(output_dim, num_hidden)
        # Learned vector standing in for the unknown label of a target token.
        self.mask_embed = nn.Parameter(t.randn(1, 1, num_hidden) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(num_hidden, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(num_hidden)
        self.readout = str(readout).lower()
        assert self.readout in ("joint", "cross"), f"bad readout {readout}"
        if self.readout == "cross":
            self.cross = nn.ModuleList([
                CrossBlock(num_hidden, n_heads, ffn_mult, dropout)
                for _ in range(n_cross_layers)])
        # The cross readout also passes the target's own acoustics to the head.
        head_in = num_hidden * (2 if self.readout == "cross" else 1)
        self.head = nn.Sequential(
            Linear(head_in, num_hidden, w_init='relu'), nn.GELU(),
            Linear(num_hidden, 2 * output_dim))
        # Variance floor (see anp.set_var_floor), not saved in the state_dict.
        self.register_buffer("var_floor", t.tensor(DEFAULT_VAR_FLOOR), persistent=False)

    def forward(self, context_x, context_y, target_x, target_y=None,
                beta: float = 1.0, predict_with_prior: bool = False,
                sensor_pos=None, sensor_mask=None,
                context_mask=None, target_mask=None):
        # beta and predict_with_prior keep the interface of the latent models
        # and have no effect. sensor_mask: optional bool (B, n_sensors), True for
        # present sensors, passed to the spatial front end.
        if self.spatial_encoder is not None:
            context_x = self.spatial_encoder(context_x, sensor_pos,
                                             sensor_mask=sensor_mask)
            target_x = self.spatial_encoder(target_x, sensor_pos,
                                            sensor_mask=sensor_mask)
        B, n = context_x.shape[0], context_x.shape[1]
        m = target_x.size(1)

        cx_e = self.x_proj(context_x)                        # acoustics only
        tx_e = self.x_proj(target_x)
        ctx = cx_e + self.y_proj(context_y)                  # acoustics + label

        km = (t.ones(B, n, dtype=t.bool, device=ctx.device)
              if context_mask is None else context_mask.to(t.bool))
        # A row with no attendable key would give NaN; keep position 0.
        km = km.clone()
        km[:, 0] |= ~km.any(dim=1)

        if self.readout == "cross":
            enc = ctx
            for blk in self.blocks:                          # context self-attention
                enc = blk(enc, key_mask=km)
            enc = self.ln_f(enc)
            q = tx_e
            for blk in self.cross:                           # q,k from acoustics; v from enc
                q = blk(q, cx_e, enc, key_mask=km)
            feat = t.cat([q, tx_e], dim=-1)
        else:
            seq = t.cat([ctx, tx_e + self.mask_embed], dim=1)
            key_mask = t.cat(
                [km, t.zeros(B, m, dtype=t.bool, device=seq.device)], dim=1)
            for blk in self.blocks:
                seq = blk(seq, key_mask=key_mask)
            feat = self.ln_f(seq[:, n:])
        mean, raw = self.head(feat).chunk(2, dim=-1)
        var = self.var_floor + F.softplus(raw)

        if target_y is not None:
            nll = reduce_nll(gaussian_nll(mean, var, None, target_y), target_mask)
            loss = nll
        else:
            nll = None
            loss = None
        return mean, var, loss, None, nll                    # kl always None

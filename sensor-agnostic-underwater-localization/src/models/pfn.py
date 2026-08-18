"""src/models/pfn.py -- Prior-Data Fitted Network backbone.

A drop-in alternative to ``anp.DeterministicModel`` / ``anp.LatentModel`` for the
SAME task (sensor_pos supplied), so it can be trained on the frozen split and
compared against CNP / ANP / RANP / spatial_CNP as one more row in the table.

WHAT IS ACTUALLY DIFFERENT (Mueller et al. ICLR'22, Hollmann et al. ICLR'23)
---------------------------------------------------------------------------
The objective the existing models minimise is ALREADY the Prior-Data NLL, so
this is not a foreign method being grafted on -- it completes one the project is
most of the way to. Three concrete changes:

1. ONE unified transformer encoder over the joint ``[context, target]``
   sequence, instead of three separate modules (LatentEncoder /
   DeterministicEncoder / Decoder) with their own projections.

2. A REAL transformer block: ``MHA -> +res -> LN -> FFN(4x) -> +res -> LN``.
   The existing ``anp.Attention`` block has **no feed-forward sublayer at all**
   -- roughly half the parameters and most of the per-token nonlinearity of a
   standard block are simply missing, and the stack is ~2 layers deep against
   TabPFN's 12. This is the cheapest of the three changes and the one most
   likely to move MAE.

3. TabPFN's attention mask (their Eq. 4): context tokens attend to each other
   bidirectionally; target tokens attend to CONTEXT ONLY -- never to each other
   and never to themselves, their own state flowing through the residual branch
   instead. One forward pass therefore emits an independent predictive for every
   target, and queries cannot leak into one another.

No positional encoding is applied on the point axis, so the model is
permutation-invariant over the context set (the correct bias for a set). There
is no latent variable and no KL term: the loss is the plain Prior-Data NLL,
which is what ``beta`` and ``predict_with_prior`` are accepted-and-ignored for.

The output head is still a diagonal Gaussian, matching the existing baselines,
so this experiment isolates the BACKBONE. Swapping in the Riemann/grid head of
spec 4.3 is a separate, later change.

INTERFACES ARE DELIBERATELY UNCHANGED -- same ``forward`` signature, same
5-tuple return ``(mean, var, loss, kl, nll)``, same ``spatial_cfg`` front end --
so this drops into the existing trainer and evaluation harness untouched.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .anp import Linear, build_spatial_encoder


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: the standard thing ``anp.Attention`` is missing.

    ``key_mask`` is (B, L) and marks which SEQUENCE POSITIONS may be used as
    keys. Because it is applied on the key axis only, masking the target region
    once makes targets invisible to every query including themselves, which is
    exactly the TabPFN rule.
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
        # (B,1,1,L) broadcasts over heads and queries: same attendable keys for all.
        am = key_mask[:, None, None, :] if key_mask is not None else None
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am,
            dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.o(a))
        x = x + self.ffn(self.ln2(x))
        return x


class CrossBlock(nn.Module):
    """Cross-attention block with separate query / key-value streams, plus FFN.

    Used by ``readout="cross"``. Queries and keys are supplied by the CALLER from
    the acoustics alone, while the values carry the full (x + y) context
    encoding -- so retrieval happens by acoustics-to-acoustics similarity and the
    label is only ever read out, never matched on.
    """

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
    """Transformer neural process with a switchable readout.

    readout="joint" (default) -- TabPFN: one stack over [context, target] with
        targets attending to context only. Faithful to the paper, but the label
        is added into the context TOKEN, so query-key matching happens in a mixed
        (acoustics + position) space. That is fine with millions of prior
        datasets; on a small training pool it measurably fails to generalise
        (topology task: train 28 / val 71, against a same-size CNP at 23 / 9).

    readout="cross" -- keep the deep FFN-equipped stack to ENCODE the context,
        then read out with CNP-style cross-attention whose queries and keys come
        from the ACOUSTICS ALONE and whose values carry the (x + y) encoding.
        This restores the retrieval bias that makes the CNP work here while
        keeping the backbone change we actually wanted to evaluate.
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
        # One learned vector standing in for "label unknown" on target tokens.
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
        # The cross readout also hands the target's own acoustics straight to the
        # head, mirroring the CNP decoder's concat of target_x with r.
        head_in = num_hidden * (2 if self.readout == "cross" else 1)
        self.head = nn.Sequential(
            Linear(head_in, num_hidden, w_init='relu'), nn.GELU(),
            Linear(num_hidden, 2 * output_dim))

    def forward(self, context_x, context_y, target_x, target_y=None,
                beta: float = 1.0, predict_with_prior: bool = False,
                sensor_pos=None, sensor_mask=None,
                context_mask=None, target_mask=None):
        # beta / predict_with_prior exist only for interface parity with the
        # latent models; a PFN has no latent, so there is nothing to weight or
        # to peek at.
        # sensor_mask: optional bool (B, n_sensors), True = present. Forwarded to
        # the shared SpatialEncoder exactly as the anp models do -- without it the
        # sensor-dropout evaluator (eval_suite tier 4) cannot score this model at
        # all, which is how the omission surfaced.
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
        # A row with no attendable key would make the softmax NaN; keep slot 0.
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
        var = 1e-3 + F.softplus(raw)

        if target_y is not None:
            nll = (0.5 * t.log(2 * t.pi * var)
                   + 0.5 * ((target_y - mean) ** 2) / var)
            if target_mask is not None:
                mm = target_mask.unsqueeze(-1)               # (B, T, 1)
                denom = mm.sum().clamp(min=1) * mean.size(-1)
                nll = (nll * mm).sum() / denom
            else:
                nll = nll.mean()
            loss = nll
        else:
            nll = None
            loss = None
        return mean, var, loss, None, nll                    # kl always None

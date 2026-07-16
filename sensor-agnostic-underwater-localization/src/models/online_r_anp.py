"""
online_r_anp.py  —  SKETCH of a streaming / online-deployable RANP
==================================================================

Why this exists
---------------
The current `r_anp.LatentModel` takes the WHOLE trajectory as `x_seq`, runs the
LSTM over it, and gathers context/target by *index into that sequence*. That is
fine for offline scoring of a recorded trajectory, but it cannot be deployed
online, where:

  * acoustic readings arrive in small CHUNKS over time (the vessel is moving),
  * we must predict the current chunk's position from what we've seen SO FAR
    (no peeking at future points),
  * we cannot re-run the encoder over the full history every step (unbounded
    compute + memory),
  * targets are *live readings*, not indices into a pre-filled tensor.

This sketch splits the model's "online state" into two pieces, each bounded:

  1. RNN state  (h_t, c_t)        -- a compact, causal summary of ALL past
                                     readings. An LSTM already gives us this for
                                     free; we just stop throwing it away.
  2. Context buffer (ring buffer) -- the last `max_context` LABELLED reference
                                     tokens (h, y) used for NP-style adaptation.
                                     Bounded => O(max_context) attention / step.

Deployment loop (pseudo)
------------------------
    state = model.init_state(batch=1, device=...)
    for x_chunk in stream:                 # (1, chunk_len, Dx) acoustic features
        mean, var, h = model.step(x_chunk, state)   # localize the chunk NOW
        emit(mean, var)                              # <- live position estimate
        # if/when ground-truth fixes arrive for some of those readings:
        if got_fix:
            model.register_context(h[:, fix_local_idx, :],
                                   y_fix, state)      # add to NP context buffer

The RNN state advances on EVERY chunk (we always have the acoustic features);
the context buffer only grows when a labelled fix becomes available.

Training alignment (the part that actually matters)
---------------------------------------------------
Online behaviour must be TRAINED, not just supported at inference. The current
trainer draws the context subset from anywhere in the sequence (including the
future) — illegal online. To train this model, run the same causal loop used at
deployment and accumulate the loss (`forward_streaming` below): for each chunk,
predict it using ONLY past context, then register that chunk's context points.
This is an autoregressive / causal Neural Process. See notes at the bottom for
the trainer + collate changes required.

NOTE: this is a sketch meant to be read and iterated on, not a drop-in trained
model. Design decisions that deserve a second pass are flagged with `DESIGN:`.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing, already-debugged building blocks.
from .r_anp import LatentEncoder, DeterministicEncoder, Decoder


# --------------------------------------------------------------------------- #
# Streaming temporal encoder: identical math to r_anp.TemporalEncoder, but the
# LSTM/GRU hidden state is THREADED THROUGH instead of discarded, so it can be
# called one chunk at a time and resumed.
# --------------------------------------------------------------------------- #
class StreamingTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0,
                 rnn_type="lstm", layer_norm=True):
        super().__init__()
        rnn_type = rnn_type.lower()
        assert rnn_type in {"lstm", "gru"}
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, dropout=rnn_dropout,
                           batch_first=True, bidirectional=False)  # causal
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_chunk: t.Tensor, rnn_state=None):
        """x_chunk: (B, L, Dx) -> h: (B, L, H), new rnn_state.
        Feeding the full sequence with rnn_state=None reproduces the offline
        TemporalEncoder exactly (causality guarantees chunked == full)."""
        h, rnn_state = self.rnn(x_chunk, rnn_state)
        h = self.norm(h)
        return h + self.input_proj(x_chunk), rnn_state


# --------------------------------------------------------------------------- #
# Bounded online state: RNN state + a ring buffer of labelled context tokens.
# Context indices are shared across the batch (same convention as the existing
# collate), so the buffers are dense (B, K, ·) tensors.
# --------------------------------------------------------------------------- #
class OnlineState:
    def __init__(self, batch_size: int, device, max_context: int):
        self.batch_size = batch_size
        self.device = device
        self.max_context = max_context
        self.rnn_state = None
        self._h: Optional[t.Tensor] = None   # (B, K, H)
        self._y: Optional[t.Tensor] = None   # (B, K, Dy)

    def append(self, h: t.Tensor, y: t.Tensor):
        """Add labelled context tokens; keep only the most recent max_context."""
        self._h = h if self._h is None else t.cat([self._h, h], dim=1)
        self._y = y if self._y is None else t.cat([self._y, y], dim=1)
        if self._h.size(1) > self.max_context:
            self._h = self._h[:, -self.max_context:, :]
            self._y = self._y[:, -self.max_context:, :]

    def context(self) -> Tuple[Optional[t.Tensor], Optional[t.Tensor]]:
        return self._h, self._y


# --------------------------------------------------------------------------- #
# Online latent RANP
# --------------------------------------------------------------------------- #
class OnlineLatentModel(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim,
                 rnn_type="lstm", rnn_layers=1, rnn_dropout=0.0, dropout=0.1,
                 max_context=128):
        super().__init__()
        self.max_context = max_context
        self.temporal_encoder = StreamingTemporalEncoder(
            input_dim=input_dim, hidden_dim=num_hidden, num_layers=rnn_layers,
            dropout=rnn_dropout, rnn_type=rnn_type, layer_norm=True)
        self.latent_encoder = LatentEncoder(
            num_hidden, num_latent=num_hidden, input_dim=num_hidden,
            output_dim=output_dim, dropout=dropout)
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_latent=num_hidden, input_dim=num_hidden,
            output_dim=output_dim, dropout=dropout)
        self.decoder = Decoder(num_hidden, input_dim=num_hidden,
                               output_dim=output_dim, dropout=dropout)

        # DESIGN: cold start. Before any fix arrives there is no context. We give
        # the encoders a single LEARNED "empty context" token so the attention /
        # pooling always has >=1 key, and the model learns a sensible
        # unconditional prior for that regime.
        self.empty_h = nn.Parameter(t.zeros(1, 1, num_hidden))
        self.empty_y = nn.Parameter(t.zeros(1, 1, output_dim))

    # ---- internal: aggregate context (with cold-start fallback) ------------ #
    def _context_tokens(self, state: OnlineState):
        ctx_h, ctx_y = state.context()
        if ctx_h is None:
            B = state.batch_size
            return (self.empty_h.expand(B, -1, -1),
                    self.empty_y.expand(B, -1, -1))
        return ctx_h, ctx_y

    # ---- internal: the ANP head (latent + deterministic + decoder) --------- #
    def _decode(self, target_h, ctx_h, ctx_y, target_y=None, predict_with_prior=False):
        """predict_with_prior: if True, the prediction uses the PRIOR latent z
        (context only) even when target_y is supplied. This is the
        deployment-faithful path: at sea the latent cannot see target labels.
        We still compute the posterior + KL/NLL when target_y is given, so the
        validation loss stays comparable to training — only the z that drives
        the prediction differs. Set False for training (posterior teacher
        forcing, lower-variance gradients)."""
        num_targets = target_h.size(1)
        prior_mu, prior_var, prior = self.latent_encoder(ctx_h, ctx_y)

        post_mu = post_var = None
        if target_y is not None:
            # Posterior sees context + this chunk's labels.
            post_mu, post_var, post = self.latent_encoder(
                t.cat([ctx_h, target_h], dim=1),
                t.cat([ctx_y, target_y], dim=1))

        use_post = (target_y is not None) and (not predict_with_prior)
        z = post if use_post else prior #type: ignore

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(ctx_h, ctx_y, target_h)
        mean, var = self.decoder(r, z, target_h)

        nll = kl = None
        if target_y is not None:
            nll = (0.5 * t.log(2 * t.pi * var)
                   + 0.5 * ((target_y - mean) ** 2) / var).mean()
            # KL normalized to per-target-point, per-dim units to match the meaned
            # NLL (see anp.py LatentModel.forward for the full rationale); beta=1
            # then reads as the standard per-point ELBO.
            kl = self._kl(prior_mu, prior_var, post_mu, post_var) \
                 / (num_targets * target_y.size(-1))
        return mean, var, kl, nll

    @staticmethod
    def _kl(prior_mu, prior_var, post_mu, post_var):  # var = log-variance
        kl = (t.exp(post_var) + (post_mu - prior_mu) ** 2) / t.exp(prior_var) \
             - 1.0 + (prior_var - post_var)
        return (0.5 * kl.sum(dim=-1)).mean()

    # ====================================================================== #
    # DEPLOYMENT API
    # ====================================================================== #
    def init_state(self, batch_size: int, device) -> OnlineState:
        return OnlineState(batch_size, device, self.max_context)

    @t.no_grad()
    def step(self, x_chunk: t.Tensor, state: OnlineState):
        """Ingest one acoustic chunk, advance the RNN, and localize it using the
        context seen so far. Returns (mean, var, h_chunk); keep h_chunk if you
        may later register some of these readings as context."""
        h, state.rnn_state = self.temporal_encoder(x_chunk, state.rnn_state)
        ctx_h, ctx_y = self._context_tokens(state)
        mean, var, _, _ = self._decode(h, ctx_h, ctx_y, target_y=None)
        return mean, var, h

    @t.no_grad()
    def register_context(self, h_tokens: t.Tensor, y_tokens: t.Tensor,
                         state: OnlineState):
        """Add labelled reference points (e.g. an arrived position fix) to the
        NP context buffer. `h_tokens` are the RNN outputs returned by `step` for
        those same readings."""
        state.append(h_tokens, y_tokens)

    # ====================================================================== #
    # TRAINING / VALIDATION:  the causal streaming loop, with gradients.
    # ====================================================================== #
    def forward_streaming(self, x_seq, y_seq, ctx_idx, chunk_size, beta=1.0,
                          predict_with_prior=False, loss_after_first_ctx_only=False,
                          detach_context=True):
        """Causal pass that mirrors deployment, used for both train and val.

        x_seq:   (B, T, Dx)   full trajectory features
        y_seq:   (B, T, Dy)   targets
        ctx_idx: 1-D LongTensor of timestep indices that become context once
                 reached (shared across the batch — same convention as the
                 offline collate). Must be causal-usable: a context point helps
                 only LATER chunks (the loop enforces past-only automatically).
        chunk_size: streaming granularity (timesteps revealed per step).
        predict_with_prior: False -> posterior teacher forcing (training);
                 True -> predict from the prior (deployment-faithful validation).

        EFFICIENCY: the RNN is run ONCE over the full sequence (causal, so the
        states are identical to streaming chunk-by-chunk); only the causal NP
        aggregation is looped over chunks on the precomputed h_seq.

        For each chunk: predict it from PAST context only, accumulate NLL/KL,
        then register the chunk's context points for future chunks.
        Returns (pred_mean, pred_var, loss, kl, nll).
        """
        B, T, _ = x_seq.shape
        device = x_seq.device

        # --- single batched RNN pass over the whole trajectory ---------------
        h_seq, _ = self.temporal_encoder(x_seq)          # (B, T, H)

        state = self.init_state(B, device)
        ctx_set = set(int(i) for i in ctx_idx.tolist())

        pred_mean = t.zeros_like(y_seq)
        pred_var = t.zeros_like(y_seq)
        tot_nll = x_seq.new_zeros(())
        tot_kl = x_seq.new_zeros(())
        n_scored = 0

        first_ctx_seen = False
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            target_h = h_seq[:, s:e]                      # gather, no re-run
            yc = y_seq[:, s:e]
            ctx_h, ctx_y = self._context_tokens(state)

            mean, var, kl, nll = self._decode(
                target_h, ctx_h, ctx_y, target_y=yc,
                predict_with_prior=predict_with_prior)
            pred_mean[:, s:e], pred_var[:, s:e] = mean, var

            # DESIGN: optionally skip loss on chunks predicted with NO real
            # context yet (pure cold-start guesses) so they don't dominate.
            if not (loss_after_first_ctx_only and not first_ctx_seen):
                tot_nll = tot_nll + nll
                tot_kl = tot_kl + kl
                n_scored += 1

            # Register this chunk's context points into the buffer (note: NOT
            # self.register_context, which is @no_grad and would cut the graph).
            local = [i for i in range(s, e) if i in ctx_set]
            if local:
                sel = t.tensor(local, device=device, dtype=t.long)
                # DESIGN: detach => truncated BPTT through the context buffer
                # (cheaper / more stable). The RNN pass still carries gradient.
                hk = h_seq[:, sel, :]
                yk = y_seq[:, sel, :]
                state.append(hk.detach() if detach_context else hk, yk)
                first_ctx_seen = True

        n_scored = max(n_scored, 1)
        nll = tot_nll / n_scored
        kl = tot_kl / n_scored
        loss = nll + beta * kl
        return pred_mean, pred_var, loss, kl, nll


# --------------------------------------------------------------------------- #
# TRAINER / COLLATE changes required to actually train this (summary)
# --------------------------------------------------------------------------- #
# 1. collate: instead of (context_x, context_y, target_x, target_y), emit
#    x_seq (B,T,Dx), y_seq (B,T,Dy), ctx_idx (timesteps that become context),
#    and a chunk_size. The temporal ORDER must be preserved (no shuffling), and
#    ctx_idx must be drawn so context is reachable causally (any timestep can be
#    context for a LATER target; the loop enforces "past-only" automatically).
# 2. model_forward dispatch: a new convention "online" -> model.forward_streaming.
# 3. eval: drive model.step / register_context over the trajectory in order and
#    score the live predictions — this measures the REAL deployment metric.
# 4. curricula worth trying: vary chunk_size and context density during training
#    so the model is robust to how fast fixes actually arrive at sea.

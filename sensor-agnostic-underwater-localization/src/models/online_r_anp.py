"""Streaming (online) version of the latent RANP.

r_anp.LatentModel processes a whole recorded trajectory at once. This model
processes acoustic readings in chunks as they arrive and keeps a bounded state:

  1. the RNN state, a causal summary of all past readings;
  2. a ring buffer with the last `max_context` labelled context tokens (h, y).

Deployment loop:
    state = model.init_state(batch=1, device=...)
    for x_chunk in stream:                         # (1, chunk_len, Dx)
        mean, var, h = model.step(x_chunk, state)  # localize the chunk
        if got_fix:                                 # a position fix arrived
            model.register_context(h[:, fix_idx, :], y_fix, state)

Training uses the same causal loop (forward_streaming): each chunk is predicted
from past context only, and its context points are added afterwards.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F

# Building blocks shared with the offline models.
from .anp import LatentEncoder, DeterministicEncoder, Decoder, gaussian_nll, gaussian_kl


# --------------------------------------------------------------------------- #
# Streaming temporal encoder: r_anp.TemporalEncoder with the RNN state passed
# in and returned, so it can be called one chunk at a time.
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
        """x_chunk: (B, L, Dx) -> h: (B, L, H), new rnn_state. With the full
        sequence and rnn_state=None this matches the offline encoder."""
        h, rnn_state = self.rnn(x_chunk, rnn_state)
        h = self.norm(h)
        return h + self.input_proj(x_chunk), rnn_state


# --------------------------------------------------------------------------- #
# Bounded online state: RNN state + a ring buffer of labelled context tokens.
# Context indices are shared across the batch, so the buffers are dense
# (B, K, .) tensors.
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
        # Attention pooling, as in the offline RANP.
        self.latent_encoder = LatentEncoder(
            num_hidden, num_latent=num_hidden, input_dim=num_hidden,
            output_dim=output_dim, dropout=dropout, pooling="attention")
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_latent=num_hidden, input_dim=num_hidden,
            output_dim=output_dim, dropout=dropout)
        self.decoder = Decoder(num_hidden, input_dim=num_hidden,
                               output_dim=output_dim, dropout=dropout)

        # Before any fix arrives there is no context: a learned "empty context"
        # token gives the attention and the pooling at least one key.
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
        """predict_with_prior: predict from the prior latent (context only) even
        when target_y is given; the posterior and the KL are still computed.
        False in training (decoding from the posterior)."""
        num_targets = target_h.size(1)
        prior_mu, prior_var, prior = self.latent_encoder(ctx_h, ctx_y)

        post_mu = post_var = None
        if target_y is not None:
            # Posterior sees context + this chunk's labels.
            post_mu, post_var, post = self.latent_encoder(
                t.cat([ctx_h, target_h], dim=1),
                t.cat([ctx_y, target_y], dim=1))

        use_post = (target_y is not None) and (not predict_with_prior)
        # Sample z in training; use the mean of the distribution in eval mode.
        if use_post:
            z = post if self.training else post_mu #type: ignore
        else:
            z = prior if self.training else prior_mu

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(ctx_h, ctx_y, target_h)
        # The decoder's correlation output is not used here (diagonal likelihood).
        mean, var, _rho = self.decoder(r, z, target_h)

        nll = kl = None
        if target_y is not None:
            nll = gaussian_nll(mean, var, None, target_y).mean()
            # KL normalized to per-point units, as in anp.LatentModel.
            kl = self._kl(prior_mu, prior_var, post_mu, post_var) \
                 / (num_targets * target_y.size(-1))
        return mean, var, kl, nll

    @staticmethod
    def _kl(prior_mu, prior_var, post_mu, post_var):  # var = log-variance
        return gaussian_kl(prior_mu, prior_var, post_mu, post_var).mean()

    # ====================================================================== #
    # Deployment API
    # ====================================================================== #
    def init_state(self, batch_size: int, device) -> OnlineState:
        return OnlineState(batch_size, device, self.max_context)

    @t.no_grad()
    def step(self, x_chunk: t.Tensor, state: OnlineState):
        """Ingest one acoustic chunk, advance the RNN and localize the chunk from
        the context seen so far. Returns (mean, var, h_chunk); keep h_chunk to
        register some of these readings as context later."""
        h, state.rnn_state = self.temporal_encoder(x_chunk, state.rnn_state)
        ctx_h, ctx_y = self._context_tokens(state)
        mean, var, _, _ = self._decode(h, ctx_h, ctx_y, target_y=None)
        return mean, var, h

    @t.no_grad()
    def register_context(self, h_tokens: t.Tensor, y_tokens: t.Tensor,
                         state: OnlineState):
        """Add labelled reference points (e.g. a position fix) to the context
        buffer. h_tokens are the RNN outputs returned by step for those readings."""
        state.append(h_tokens, y_tokens)

    # ====================================================================== #
    # Training and validation: the causal streaming loop
    # ====================================================================== #
    def forward_streaming(self, x_seq, y_seq, ctx_idx, chunk_size, beta=1.0,
                          predict_with_prior=False, loss_after_first_ctx_only=False,
                          detach_context=True):
        """Causal pass that mirrors deployment, used in training and validation.

        x_seq:   (B, T, Dx) trajectory features
        y_seq:   (B, T, Dy) targets
        ctx_idx: 1-D LongTensor of timesteps that become context once reached
                 (shared across the batch)
        chunk_size: timesteps revealed per step
        predict_with_prior: False in training, True for validation

        The RNN runs once over the full sequence (it is causal, so the states
        are the same as when streaming); only the context aggregation is looped
        over chunks. Returns (pred_mean, pred_var, loss, kl, nll).
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

            # Optionally skip the loss on chunks predicted with no context yet.
            if not (loss_after_first_ctx_only and not first_ctx_seen):
                tot_nll = tot_nll + nll
                tot_kl = tot_kl + kl
                n_scored += 1

            # Add this chunk's context points to the buffer (register_context
            # is no_grad, so it is not used here).
            local = [i for i in range(s, e) if i in ctx_set]
            if local:
                sel = t.tensor(local, device=device, dtype=t.long)
                # Detached: truncated backpropagation through the buffer.
                hk = h_seq[:, sel, :]
                yk = y_seq[:, sel, :]
                state.append(hk.detach() if detach_context else hk, yk)
                first_ctx_seen = True

        n_scored = max(n_scored, 1)
        nll = tot_nll / n_scored
        kl = tot_kl / n_scored
        loss = nll + beta * kl
        return pred_mean, pred_var, loss, kl, nll



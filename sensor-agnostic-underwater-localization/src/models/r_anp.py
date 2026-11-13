"""Recurrent neural processes.

LatentModel (RANP) and DeterministicModel (RCNP) run a causal LSTM/GRU over the
whole trajectory, take the context and target points from its hidden states by
index, and pass them to the attentive encoders and decoders of anp.py.

Forward convention (differs from anp.py):
    forward(x_seq, context_indices, context_y, target_indices, target_y, beta)
where x_seq is the full (B, T, Dx) sequence.

The RANP latent encoder pools with learned attention weights by default (the
ANP uses a plain mean), since its inputs are RNN states rather than independent
samples. latent_pooling switches it.
"""

import torch as t
import torch.nn as nn

from .anp import (LatentEncoder as _LatentEncoder, DeterministicEncoder, Decoder,
                  DeterministicDecoder, build_spatial_encoder, gaussian_nll,
                  gaussian_kl)


class TemporalEncoder(nn.Module):
    """
    Causal temporal encoder (LSTM/GRU) for sequences.
    Input:  x_seq (B, T, Dx)
    Output: h_seq (B, T, Dh)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "lstm", # "lstm" | "gru"
        layer_norm: bool = True,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        assert rnn_type in {"lstm", "gru"}

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False, # causal (no future leakage)
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )

        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_seq: t.Tensor) -> t.Tensor:
        # x_seq: (B, T, Dx)

        h_seq, _ = self.rnn(x_seq)      # (B, T, Dh)
        h_seq = self.norm(h_seq)
        return h_seq + self.input_proj(x_seq)


class LatentModel(nn.Module):
    """RANP: recurrent encoder followed by attentive latent and deterministic
    paths."""
    def __init__(
        self,
        num_hidden: int,
        input_dim: int,        # dimension of x_seq before the RNN
        output_dim: int,
        rnn_type: str = "lstm",
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        dropout: float = 0.1,  # dropout for attention + decoder MLP
        spatial_cfg=None,      # optional spatial front end
        full_cov: bool = False,  # correlated (x, y) output covariance
        attn_ffn: bool = False,  # feed-forward sublayer in the attention blocks
        latent_pooling: str = "attention",  # see the module docstring
        ):
        super(LatentModel, self).__init__()
        # Optional spatial front end before the RNN: (B, T, feat) -> (B, T, num_hidden).
        self.spatial_encoder, temporal_in = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        # RNN: temporal_in -> num_hidden
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_in,
            hidden_dim=num_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            rnn_type=rnn_type,
            layer_norm=True,
        )

        # The NP encoders take the RNN states (num_hidden) as their inputs.
        self.latent_encoder = _LatentEncoder(
            num_hidden, num_latent=num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
            attn_ffn=attn_ffn,
            pooling=latent_pooling,
        )
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_latent=num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
            attn_ffn=attn_ffn,
        )
        self.decoder = Decoder(
            num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
            full_cov=full_cov,
        )

    def forward(
        self,
        x_seq: t.Tensor, # (B, T, Dx) full sequence
        context_indices: t.Tensor, # (Nc,) context indices
        context_y: t.Tensor, # (B, Nc, output_dim)
        target_indices: t.Tensor, # (Nt,) target indices (usually 0..T-1)
        target_y: t.Tensor = None,  #type: ignore[assignment] # (B, Nt, output_dim); None at inference
        beta: float = 1.0,
        predict_with_prior: bool = False,
        sensor_pos=None,           # (B, n_sensors, 3); needed by the spatial front end
        sensor_mask=None,          # (B, n_sensors) bool, True for present sensors
        ):
        # predict_with_prior: decode from the prior latent (context only) even
        # when target_y is given; the posterior and the KL are still computed.
        if self.spatial_encoder is not None:
            x_seq = self.spatial_encoder(x_seq, sensor_pos, sensor_mask=sensor_mask)
        # The RNN runs over the whole sequence, inside the model.
        h_seq = self.temporal_encoder(x_seq) # (B, T, num_hidden)

        # Split context / target from the RNN hidden states.
        context_x = h_seq[:, context_indices, :] # (B, Nc, H)
        target_x  = h_seq[:, target_indices,  :] # (B, Nt, H)
        num_targets = target_x.size(1)

        # Latent path
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        posterior_mu = posterior_var = None
        if target_y is not None:
            # The posterior conditions on context and targets (training only).
            posterior_mu, posterior_var, posterior = self.latent_encoder(
                t.cat([context_x, target_x], dim=1),
                t.cat([context_y, target_y], dim=1))

        use_posterior = (target_y is not None) and (not predict_with_prior)
        # Sample z in training; use the mean of the distribution in eval mode.
        if use_posterior:
            z = posterior if self.training else posterior_mu
        else:
            z = prior if self.training else prior_mu

        z = z.unsqueeze(1).repeat(1, num_targets, 1)

        # Deterministic path (cross-attention)
        r = self.deterministic_encoder(context_x, context_y, target_x)

        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            nll = nll.mean()
            # KL normalized to per-point units, as in anp.LatentModel.
            kl = gaussian_kl(prior_mu, prior_var, posterior_mu, posterior_var).mean()
            kl = kl / (num_targets * target_y.size(-1))
            loss = nll + beta * kl
        else:
            kl = loss = nll = None

        return y_pred_mean, y_pred_var, loss, kl, nll

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        # prior_var / posterior_var are log-variances; batch-mean KL (scalar).
        return gaussian_kl(prior_mu, prior_var, posterior_mu, posterior_var).mean()


class DeterministicModel(nn.Module):
    """RCNP: recurrent encoder followed by an attentive deterministic encoder and decoder."""
    def __init__(
        self,
        num_hidden: int,
        input_dim: int,
        output_dim: int,
        rnn_type: str = "lstm",
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        dropout: float = 0.1,  # dropout for attention + decoder MLP
        spatial_cfg=None,      # optional spatial front end
        full_cov: bool = False,  # correlated (x, y) output covariance
        attn_ffn: bool = False,  # feed-forward sublayer in the attention blocks
    ):
        super(DeterministicModel, self).__init__()
        self.spatial_encoder, temporal_in = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_in,
            hidden_dim=num_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            rnn_type=rnn_type,
            layer_norm=True,
        )
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_latent=num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
            attn_ffn=attn_ffn,
        )
        self.decoder = DeterministicDecoder(
            num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
            full_cov=full_cov,
        )

    def forward(
        self,
        x_seq: t.Tensor,
        context_indices: t.Tensor,
        context_y: t.Tensor,
        target_indices: t.Tensor,
        target_y: t.Tensor = None, #type: ignore[assignment]
        beta: float = 1.0,
        predict_with_prior: bool = False,
        sensor_pos=None,           # (B, n_sensors, 3); needed by the spatial front end
        sensor_mask=None,          # (B, n_sensors) bool, True for present sensors
    ):
        # predict_with_prior keeps the interface of the RANP and has no effect.
        if self.spatial_encoder is not None:
            x_seq = self.spatial_encoder(x_seq, sensor_pos, sensor_mask=sensor_mask)
        h_seq = self.temporal_encoder(x_seq)
        context_x = h_seq[:, context_indices, :]
        target_x  = h_seq[:, target_indices,  :]

        r = self.deterministic_encoder(context_x, context_y, target_x)
        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            nll = nll.mean()
            loss = nll
        else:
            nll = None
            loss = None
        return y_pred_mean, y_pred_var, loss, None, nll  # kl always None

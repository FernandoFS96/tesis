import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .anp import gaussian_nll
# Reuse the sensor-position-aware front end from the offline models so the
# recurrent variants (spatial_ranp / spatial_rcnp) get the same displacement-
# robust per-point embedding before the RNN.
from src.models.anp import build_spatial_encoder

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init)) #type: ignore[no-untyped-call]

    def forward(self, x):
        return self.linear_layer(x)

# ─────────────────────────────────────────────────────────────────────────────
# LatentEncoder: sustituir mean-pooling por last-step pooling
#
# ANTES: hidden = encoder_input.mean(dim=1)
#
# PROBLEMA: cuando la entrada ya son estados ocultos del RNN (h_t), el promedio uniforme destruye la estructura temporal acumulada por el RNN. 
# El último estado h_{T_ctx} ya contiene un resumen causal de todo el contexto.
#
# AHORA: se toma el último paso temporal del contexto como representación global.
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_nll(mean, var, rho, y):
    """Per-element Gaussian NLL, shape (..., D).

    rho is None -> exactly the independent diagonal Gaussian these models have
    always used (bit-identical, not merely equivalent).

    rho given -> the FIRST TWO output dims (x, y) form a CORRELATED 2x2 block and
    any remaining dims stay independent. z is deliberately excluded: it is
    constant in this data (y_std[2] ~ 1e-6), so a full 3x3 covariance would be
    singular. Measured motivation: the true posterior on a random array has
    |corr| ~ 0.71, and the best possible AXIS-ALIGNED Gaussian is 0.74 nats from
    it while a correlated one is 0.15 -- a gap no amount of training can close
    with a diagonal head.

    The joint (x, y) term is split evenly across the two slots so the returned
    shape, and therefore every existing .mean() / target-mask reduction
    downstream, is unchanged. Only the SUM has meaning, and it is correct.
    """
    d = y - mean
    if rho is None:
        return 0.5 * t.log(2 * t.pi * var) + 0.5 * (d ** 2) / var
    out = 0.5 * t.log(2 * t.pi * var) + 0.5 * (d ** 2) / var
    r = rho[..., 0] if rho.shape[-1] == 1 else rho
    vx, vy = var[..., 0], var[..., 1]
    dx, dy = d[..., 0], d[..., 1]
    om = (1.0 - r * r).clamp(min=1e-6)
    q = (dx * dx / vx - 2 * r * dx * dy / t.sqrt(vx * vy) + dy * dy / vy) / om
    nll_xy = 0.5 * (q + t.log(om) + t.log(vx) + t.log(vy)) + t.log(t.tensor(2 * t.pi))
    out = out.clone()
    out[..., 0] = 0.5 * nll_xy
    out[..., 1] = 0.5 * nll_xy
    return out


class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1, attn_ffn=False):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout, ffn=attn_ffn) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_var = Linear(num_hidden, num_latent)
        self.pool_attn = nn.Linear(num_hidden, 1)

    def forward(self, x, y):
        encoder_input = t.cat([x, y], dim=-1)
        encoder_input = self.input_projection(encoder_input)
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        #hidden = encoder_input[:, -1, :]
        scores = self.pool_attn(encoder_input)          # (B, T, 1)
        weights = t.softmax(scores, dim=1)          # (B, T, 1)
        hidden = (encoder_input * weights).sum(dim=1)   # (B, H)
        hidden = t.relu(self.penultimate_layer(hidden))
        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        log_var = 3 * t.tanh(log_var)

        std = t.exp(0.5 * log_var)
        std = t.clamp(std, min=1e-6, max=1e6)
        eps = t.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, log_var, z

class DeterministicEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1, attn_ffn=False):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout, ffn=attn_ffn) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout, ffn=attn_ffn) for _ in range(2)])
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.context_projection = Linear(input_dim, num_hidden)
        self.target_projection = Linear(input_dim, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = t.cat([context_x, context_y], dim=-1)
        encoder_input = self.input_projection(encoder_input)

        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query

class Decoder(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1,
                 full_cov=False):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        # Per-layer LayerNorm + dropout in the decoder MLP.
        self.norms = nn.ModuleList([nn.LayerNorm(num_hidden * 3) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.mean_projection = Linear(num_hidden*3, output_dim)
        self.log_var_projection = Linear(num_hidden*3, output_dim)
        # x-y correlation; absent entirely when full_cov=False, so the
        # state_dict is unchanged and old checkpoints still load.
        self.rho_projection = Linear(num_hidden*3, 1) if full_cov else None


    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        target_x = self.target_projection(target_x)
        hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)
        for linear, norm in zip(self.linears, self.norms):
            hidden = t.relu(linear(hidden))
            hidden = norm(hidden)
            hidden = self.dropout(hidden)
        mean = self.mean_projection(hidden)
        var = 1e-3 + F.softplus(self.log_var_projection(hidden))
        rho = (0.99 * t.tanh(self.rho_projection(hidden))
               if self.rho_projection is not None else None)
        return mean, var, rho

class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_k, dropout=0.1):
        super().__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, key, value, query):
        """
        key, value: (B', Lk, d)
        query:      (B', Lq, d)
        """
        # SDPA apply scale 1/sqrt(d) internally and can use optimized kernels
        dropout_p = self.attn_dropout.p if self.training else 0.0

        # scaled_dot_product_attention expects (query, key, value)
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False
        )

        # For max perf, SDPA does not return attention weights.
        attn_weights = None
        return out, attn_weights


class Attention(nn.Module):
    """Recurrent-model attention sublayer, optionally with a feed-forward stage.

    This is a SEPARATE class from anp.Attention (the recurrent models keep their
    own, with no key_padding_mask), so the ffn retrofit has to be applied here
    too or spatial_ranp / spatial_rcnp would silently keep the old block while
    the CNP/ANP got the fix. Semantics match anp.Attention exactly: ffn=False is
    bit-identical to the original and creates no parameters.
    """

    def __init__(self, num_hidden, h=4, dropout=0.1, ffn=False, ffn_mult=4):
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout=dropout)
        self.residual_dropout = nn.Dropout(p=dropout)
        self.final_linear = Linear(num_hidden * 2, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)
        if ffn:
            self.ffn_norm = nn.LayerNorm(num_hidden)
            self.ffn = nn.Sequential(
                Linear(num_hidden, ffn_mult * num_hidden, w_init='relu'),
                nn.GELU(),
                nn.Dropout(dropout),
                Linear(ffn_mult * num_hidden, num_hidden),
            )
        else:
            self.ffn_norm = None
            self.ffn = None

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        result, attns = self.multihead(key, value, query)

        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        result = t.cat([residual, result], dim=-1)
        result = self.final_linear(result)
        result = self.residual_dropout(result)
        result = result + residual
        result = self.layer_norm(result)
        if self.ffn is not None:
            result = result + self.ffn(self.ffn_norm(result))
        return result, attns

# TemporalEncoder: Causal temporal encoder (LSTM/GRU) for sequences.
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

# ─────────────────────────────────────────────────────────────────────────────
# LatentModel: integra TemporalEncoder como componente interno
#
# ANTES: El TemporalEncoder vivía fuera, gestionado manualmente en el script de entrenamiento. Esto implicaba:
#        - Que importar LatentModel de r_anp.py daba un ANP puro (sin RNN)
#        - Gestión externa del estado train/eval del rnn_encoder
#        - Gradient clipping aplicado solo a model.parameters(), sin el RNN
#
# AHORA: LatentModel contiene el TemporalEncoder. El forward recibe x_seq completo (B, T, Dx) y los índices de contexto/target, 
#        aplica el RNN internamente, y hace el split dentro del modelo.
#
# FIRMA DEL FORWARD CAMBIA:
#   ANTES: forward(context_x, context_y, target_x, target_y, beta)
#   AHORA: forward(x_seq, context_indices, context_y, target_indices, target_y, beta)
# ─────────────────────────────────────────────────────────────────────────────
class LatentModel(nn.Module):
    def __init__(
        self,
        num_hidden: int,
        input_dim: int,        # dimensión de x_seq ANTES del RNN (Dx+S)
        output_dim: int,
        rnn_type: str = "lstm",
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        dropout: float = 0.1,  # dropout for attention + decoder MLP
        spatial_cfg=None,      # sensor-position-aware front end (spatial_ranp)
        full_cov: bool = False,  # correlated (x, y) output covariance
        attn_ffn: bool = False,  # feed-forward sublayer in the attention blocks
        ):
        super(LatentModel, self).__init__()
        # Optional spatial front end BEFORE the RNN: maps raw (B,T,feat) acoustics
        # to (B,T,num_hidden). When enabled the RNN input_dim becomes num_hidden.
        self.spatial_encoder, temporal_in = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        # RNN integrado: temporal_in → num_hidden
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_in,
            hidden_dim=num_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            rnn_type=rnn_type,
            layer_norm=True,
        )

        # El input de los encoders ANP ahora es num_hidden (salida del RNN)
        self.latent_encoder = LatentEncoder(
            num_hidden, num_latent=num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
            attn_ffn=attn_ffn,
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
        x_seq: t.Tensor, # (B, T, Dx+S) — secuencia completa aumentada
        context_indices: t.Tensor, # (Nc,)  índices del contexto
        context_y: t.Tensor, # (B, Nc, output_dim)
        target_indices: t.Tensor, # (Nt,)  índices de los targets (normalmente 0..T-1)
        target_y: t.Tensor = None,  #type: ignore[no-untyped-call] # (B, Nt, output_dim) — None en inferencia
        beta: float = 1.0,
        predict_with_prior: bool = False,
        sensor_pos=None,           # (B, n_sensors, 3); required iff spatial front end on
        sensor_mask=None,          # (B, n_sensors) bool; drops failed sensors (spatial only)
        ):
        # predict_with_prior: if True the DECODER is driven by the PRIOR latent
        # (context only) even when target_y is supplied, deployment-faithful
        # inference. Posterior + KL/NLL are still computed for logging; only the z
        # driving the prediction changes. False = training (teacher forcing).
        # Spatial front end (if any) maps raw acoustics to per-point embeddings
        # BEFORE the RNN; the LSTM then integrates them causally over time.
        if self.spatial_encoder is not None:
            x_seq = self.spatial_encoder(x_seq, sensor_pos, sensor_mask=sensor_mask)
        # RNN aplicado internamente sobre la secuencia completa
        h_seq = self.temporal_encoder(x_seq) # (B, T, num_hidden)

        # Split context / target sobre los estados ocultos del RNN
        context_x = h_seq[:, context_indices, :] # (B, Nc, H)
        target_x  = h_seq[:, target_indices,  :] # (B, Nt, H)
        num_targets = target_x.size(1)

        # Camino latente
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        posterior_mu = posterior_var = None
        if target_y is not None:
            # Posterior conditions on context UNION targets (standard ANP; see
            # anp.py LatentModel.forward for the rationale). Training-only.
            posterior_mu, posterior_var, posterior = self.latent_encoder(
                t.cat([context_x, target_x], dim=1),
                t.cat([context_y, target_y], dim=1))

        use_posterior = (target_y is not None) and (not predict_with_prior)
        # train(): decode a SAMPLE of z; eval(): decode the distribution MEAN
        # (deterministic point prediction; see anp.py for the rationale).
        if use_posterior:
            z = posterior if self.training else posterior_mu
        else:
            z = prior if self.training else prior_mu

        z = z.unsqueeze(1).repeat(1, num_targets, 1)

        # Camino determinista (cross-attention)
        r = self.deterministic_encoder(context_x, context_y, target_x)

        # Decoder
        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            nll = nll.mean()
            # KL normalized to per-target-point, per-dim units to match the meaned
            # NLL (see anp.py LatentModel.forward for the full rationale); beta=1
            # then reads as the standard ELBO instead of ~num_targets*output_dim.
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var) #type: ignore[assignment]
            kl = kl / (num_targets * target_y.size(-1))
            loss = nll + beta * kl
        else:
            kl = loss = nll = None

        return y_pred_mean, y_pred_var, loss, kl, nll

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        # prior_var / posterior_var are log-variances
        kl = (t.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / t.exp(prior_var) - 1.0 \
             + (prior_var - posterior_var) # (B, latent_dim)
        kl = 0.5 * kl.sum(dim=-1) # (B,)
        return kl.mean() # scalar, stable vs batch size


class DeterministicDecoder(nn.Module):
    """Decoder for RCNP: takes only the deterministic representation r (no latent z)."""
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1,
                 full_cov=False):
        super(DeterministicDecoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 2, num_hidden * 2, w_init='relu') for _ in range(3)])
        # Per-layer LayerNorm + dropout in the decoder MLP.
        self.norms = nn.ModuleList([nn.LayerNorm(num_hidden * 2) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.mean_projection = Linear(num_hidden * 2, output_dim)
        self.log_var_projection = Linear(num_hidden * 2, output_dim)
        self.rho_projection = Linear(num_hidden * 2, 1) if full_cov else None

    def forward(self, r, target_x):
        target_x = self.target_projection(target_x)
        hidden = t.cat([r, target_x], dim=-1)
        for linear, norm in zip(self.linears, self.norms):
            hidden = t.relu(linear(hidden))
            hidden = norm(hidden)
            hidden = self.dropout(hidden)
        mean = self.mean_projection(hidden)
        var = 1e-3 + F.softplus(self.log_var_projection(hidden))
        rho = (0.99 * t.tanh(self.rho_projection(hidden))
               if self.rho_projection is not None else None)
        return mean, var, rho


class DeterministicModel(nn.Module):
    """RCNP: RNN temporal encoder + attentive deterministic encoder + decoder, no latent variable."""
    def __init__(
        self,
        num_hidden: int,
        input_dim: int,
        output_dim: int,
        rnn_type: str = "lstm",
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        dropout: float = 0.1,  # dropout for attention + decoder MLP
        spatial_cfg=None,      # sensor-position-aware front end (spatial_rcnp)
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
        sensor_pos=None,           # (B, n_sensors, 3); required iff spatial front end on
        sensor_mask=None,          # (B, n_sensors) bool; drops failed sensors (spatial only)
    ):
        # predict_with_prior accepted for interface parity with the latent RANP;
        # a deterministic RCNP never peeks at target labels, so it has no effect.
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
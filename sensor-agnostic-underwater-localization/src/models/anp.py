import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
# src/models/anp.py: Define ANP components

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))  # type: ignore[arg-type]

    def forward(self, x):
        return self.linear_layer(x)

class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_var = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        encoder_input = t.cat([x, y], dim=-1)
        encoder_input = self.input_projection(encoder_input)
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        hidden = encoder_input.mean(dim=1)
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
    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout) for _ in range(2)])
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
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        # Legacy decoder body: plain relu(Linear(...)) stack (no LayerNorm/dropout).
        self.mean_projection = Linear(num_hidden*3, output_dim)
        self.log_var_projection = Linear(num_hidden*3, output_dim)


    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        target_x = self.target_projection(target_x)
        hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)
        for linear in self.linears:
            hidden = t.relu(linear(hidden))
        mean = self.mean_projection(hidden)
        var = 1e-3 + F.softplus(self.log_var_projection(hidden))
        return mean, var

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
    def __init__(self, num_hidden, h=4, dropout=0.1):
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
        return result, attns


# =========================================================================== #
# Spatial (sensor-position-aware) front-end encoder
# =========================================================================== #
class FourierPositionEncoding(nn.Module):
    """Map a low-dim physical position to a higher-dim embedding via sinusoids at
    several spatial wavelengths (à la NeRF / PEACH). Wavelengths are in PHYSICAL
    metres (log-spaced over [min_wavelength, max_wavelength]) so positions can be
    fed in their absolute frame, do NOT centre/normalise per-geometry, or the
    sensor-DISPLACEMENT signal we are trying to encode is erased.

    Output dim = in_dim * 2 * n_bands (sin & cos per band per coordinate)."""

    def __init__(self, n_bands=8, min_wavelength=10.0, max_wavelength=1000.0, in_dim=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim * 2 * n_bands
        wavelengths = t.logspace(math.log10(min_wavelength),
                                 math.log10(max_wavelength), n_bands)
        freqs = 2.0 * math.pi / wavelengths            # (n_bands,)
        self.register_buffer("freqs", freqs)

    def forward(self, pos):
        # pos: (..., in_dim) in physical metres.
        proj = pos[..., None] * self.freqs             # (..., in_dim, n_bands)
        emb = t.cat([t.sin(proj), t.cos(proj)], dim=-1)  # (..., in_dim, 2*n_bands)
        return emb.flatten(-2)                          # (..., in_dim*2*n_bands)


class SpatialEncoder(nn.Module):
    """Sensor-position-aware front end. Turns each trajectory point's flat
    ``feat_dim = tau * n_sensors`` acoustic vector into a per-point ``num_hidden``
    embedding that is *permutation-equivariant over sensors* and tagged with each
    sensor's physical position, so a displaced layout is a new set of
    (position, measurement) pairs rather than a scrambled vector.

    Pipeline (tokenize=True): per-sensor tokens -> concat Fourier position ->
    shared projection -> [self-attention across sensors] -> pool -> per-point vec.

    Flags expose the ablation ladder:
      tokenize      False -> keep the flat vector, only (optionally) append a
                             flattened position embedding, then project (the
                             "does it just need position?" control, NOT
                             permutation invariant).
      use_position  add Fourier position features (else pure acoustics).
      use_attention cross-sensor self-attention (else Deep-Sets pool only).
      pooling       'attention' (learned query) | 'mean'.

    IMPORTANT reshape: the flat vector is tau-major / sensor-minor
    (flat[k] = feature(tau = k // n_sensors, sensor = k % n_sensors)), so
    ``x.view(..., tau, n_sensors)`` then move the sensor axis. A naive
    ``view(..., n_sensors, tau)`` would mix sensors and silently break everything.
    """

    def __init__(self, feat_dim, n_sensors, num_hidden, *, tokenize=True,
                 use_position=True, use_attention=True, n_attn_layers=1,
                 pooling="attention", n_fourier_bands=8, min_wavelength=10.0,
                 max_wavelength=1000.0, pos_dim=2, dropout=0.1, norm_acoustic=True):
        super().__init__()
        assert feat_dim % n_sensors == 0, \
            f"feat_dim {feat_dim} not divisible by n_sensors {n_sensors}"
        self.feat_dim = feat_dim
        self.n_sensors = n_sensors
        self.tau = feat_dim // n_sensors
        self.tokenize = tokenize
        self.use_position = use_position
        self.use_attention = use_attention and tokenize
        self.pooling = pooling
        self.pos_dim = pos_dim
        self.num_hidden = num_hidden

        self.pos_enc = (FourierPositionEncoding(n_fourier_bands, min_wavelength,
                                                max_wavelength, in_dim=pos_dim)
                        if use_position else None)
        pos_out = self.pos_enc.out_dim if use_position else 0

        # CRITICAL: the raw acoustic features are ~2 orders of magnitude smaller
        # (std ~5e-3) than the unit-scale Fourier position features, so without
        # this the position swamps the acoustics in the token and the model
        # collapses to constant prediction. LayerNorm brings the acoustic token to
        # unit scale so it competes with position from step 1.
        if tokenize:
            self.acoustic_norm = nn.LayerNorm(self.tau) if norm_acoustic else None
            # per-sensor token = (tau acoustic) [+ position embedding], shared proj
            self.token_proj = Linear(self.tau + pos_out, num_hidden, w_init='relu')
            if self.use_attention:
                self.attn_layers = nn.ModuleList(
                    [Attention(num_hidden, dropout=dropout) for _ in range(n_attn_layers)])
            if pooling == "attention":
                self.pool_query = nn.Parameter(t.randn(1, 1, num_hidden) * 0.02)
                self.pool_attn = Attention(num_hidden, dropout=dropout)
        else:
            self.acoustic_norm = nn.LayerNorm(feat_dim) if norm_acoustic else None
            # flat control: raw feat vector [+ all sensors' position embeddings]
            self.flat_proj = Linear(feat_dim + n_sensors * pos_out, num_hidden,
                                    w_init='relu')

    def forward(self, x, sensor_pos):
        # x: (B, N, feat_dim);  sensor_pos: (B, n_sensors, 3), constant over the
        # N points of a trajectory (same geometry). Returns (B, N, num_hidden).
        B, N, F = x.shape
        pos = sensor_pos[..., :self.pos_dim] if self.use_position else None  # (B,S,pd)

        if not self.tokenize:
            xa = self.acoustic_norm(x) if self.acoustic_norm is not None else x
            if self.use_position:
                pe = self.pos_enc(pos).reshape(B, -1)          # (B, S*pos_out)
                pe = pe[:, None, :].expand(B, N, -1)           # broadcast over points
                xa = t.cat([xa, pe], dim=-1)
            return t.relu(self.flat_proj(xa))

        # tau-major / sensor-minor -> (B, N, tau, S) -> (B, N, S, tau)
        tok = x.view(B, N, self.tau, self.n_sensors).transpose(-1, -2)
        if self.acoustic_norm is not None:
            tok = self.acoustic_norm(tok)                     # balance acoustic vs position scale
        if self.use_position:
            pe = self.pos_enc(pos)                             # (B, S, pos_out)
            pe = pe[:, None, :, :].expand(B, N, self.n_sensors, -1)
            tok = t.cat([tok, pe], dim=-1)                     # (B, N, S, tau+pos_out)
        tok = self.token_proj(tok)                            # (B, N, S, H)

        tok = tok.reshape(B * N, self.n_sensors, self.num_hidden)
        if self.use_attention:
            for attn in self.attn_layers:
                tok, _ = attn(tok, tok, tok)                   # self-attn over sensors
        if self.pooling == "attention":
            q = self.pool_query.expand(B * N, -1, -1)          # (B*N, 1, H)
            pooled, _ = self.pool_attn(tok, tok, q)            # query attends sensors
            pooled = pooled.squeeze(1)
        else:
            pooled = tok.mean(dim=1)                           # Deep-Sets mean pool
        return pooled.view(B, N, self.num_hidden)


def build_spatial_encoder(spatial_cfg, feat_dim, num_hidden, dropout):
    """Instantiate a SpatialEncoder from a config mapping, or None if disabled.
    Returns (encoder_or_None, encoder_output_dim). When enabled the downstream NP
    encoders take ``num_hidden`` as their input_dim instead of the raw feat_dim."""
    if not spatial_cfg:
        return None, feat_dim
    get = spatial_cfg.get
    if not bool(get("enabled", False)):
        return None, feat_dim
    enc = SpatialEncoder(
        feat_dim=feat_dim, n_sensors=int(get("n_sensors", 10)), num_hidden=num_hidden,
        tokenize=bool(get("tokenize", True)),
        use_position=bool(get("use_position", True)),
        use_attention=bool(get("use_attention", True)),
        n_attn_layers=int(get("n_attn_layers", 1)),
        pooling=str(get("pooling", "attention")),
        n_fourier_bands=int(get("n_fourier_bands", 8)),
        min_wavelength=float(get("min_wavelength", 10.0)),
        max_wavelength=float(get("max_wavelength", 1000.0)),
        pos_dim=int(get("pos_dim", 2)),
        norm_acoustic=bool(get("norm_acoustic", True)), dropout=dropout)
    return enc, num_hidden


# LatentModel:

class LatentModel(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1, spatial_cfg=None):
        super(LatentModel, self).__init__()
        # Optional sensor-position-aware front end. When enabled it maps the raw
        # (B, N, feat_dim) acoustics to (B, N, num_hidden), so the NP encoders see
        # num_hidden-dim inputs instead of the flat feat_dim.
        self.spatial_encoder, enc_input_dim = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.latent_encoder = LatentEncoder(num_hidden, num_latent=num_hidden,
                                            input_dim=enc_input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout)
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=enc_input_dim,
                                                          output_dim=output_dim,
                                                          dropout=dropout)
        self.decoder = Decoder(num_hidden,
                               input_dim=enc_input_dim,
                               output_dim=output_dim,
                               dropout=dropout)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0,
                predict_with_prior: bool = False, sensor_pos=None):
        # predict_with_prior: if True the DECODER is driven by the PRIOR latent
        # (context only) even when target_y is supplied, the deployment-faithful
        # path (at inference the latent cannot peek at target labels). The
        # posterior + KL/NLL are still computed when target_y is given, so the
        # validation loss stays comparable to training; only the z that drives the
        # prediction changes. Use False for training (posterior teacher forcing).
        if self.spatial_encoder is not None:
            # sensor_pos is per-trajectory (same geometry for context & targets).
            context_x = self.spatial_encoder(context_x, sensor_pos)
            target_x = self.spatial_encoder(target_x, sensor_pos)
        num_targets = target_x.size(1)
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        posterior_mu = posterior_var = None
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)

        use_posterior = (target_y is not None) and (not predict_with_prior)
        z = posterior if use_posterior else prior

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(context_x, context_y, target_x)

        y_pred_mean, y_pred_var = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = 0.5 * t.log(2 * t.pi * y_pred_var) + 0.5 * ((target_y - y_pred_mean) ** 2) / y_pred_var
            nll = nll.mean()
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var) #type: ignore[assignment]
            loss = nll + beta * kl
        else:
            kl = None
            loss = None
            nll = None
        return y_pred_mean, y_pred_var, loss, kl, nll

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        # prior_var / posterior_var are log-variances
        kl = (t.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / t.exp(prior_var) - 1.0 \
             + (prior_var - posterior_var) # (B, latent_dim)
        kl = 0.5 * kl.sum(dim=-1) # (B,)
        return kl.mean() # scalar, stable vs batch size


class DeterministicDecoder(nn.Module):
    """Decoder for CNP: takes only the deterministic representation r (no latent z)."""
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1):
        super(DeterministicDecoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 2, num_hidden * 2, w_init='relu') for _ in range(3)])
        # Per-layer LayerNorm + dropout in the decoder MLP.
        self.norms = nn.ModuleList([nn.LayerNorm(num_hidden * 2) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.mean_projection = Linear(num_hidden * 2, output_dim)
        self.log_var_projection = Linear(num_hidden * 2, output_dim)

    def forward(self, r, target_x):
        target_x = self.target_projection(target_x)
        hidden = t.cat([r, target_x], dim=-1)
        for linear, norm in zip(self.linears, self.norms):
            hidden = t.relu(linear(hidden))
            hidden = norm(hidden)
            hidden = self.dropout(hidden)
        mean = self.mean_projection(hidden)
        var = 1e-3 + F.softplus(self.log_var_projection(hidden))
        return mean, var


class DeterministicModel(nn.Module):
    """CNP: attentive deterministic encoder + decoder, no latent variable."""
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1, spatial_cfg=None):
        super(DeterministicModel, self).__init__()
        self.spatial_encoder, enc_input_dim = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=enc_input_dim,
                                                          output_dim=output_dim,
                                                          dropout=dropout)
        self.decoder = DeterministicDecoder(num_hidden,
                                            input_dim=enc_input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0,
                predict_with_prior: bool = False, sensor_pos=None):
        # predict_with_prior is accepted for a uniform interface with the latent
        # models but has no effect: a deterministic CNP never uses target labels
        # to predict, so there is no posterior to peek at.
        if self.spatial_encoder is not None:
            context_x = self.spatial_encoder(context_x, sensor_pos)
            target_x = self.spatial_encoder(target_x, sensor_pos)
        r = self.deterministic_encoder(context_x, context_y, target_x)
        y_pred_mean, y_pred_var = self.decoder(r, target_x)

        if target_y is not None:
            nll = 0.5 * t.log(2 * t.pi * y_pred_var) + 0.5 * ((target_y - y_pred_mean) ** 2) / y_pred_var
            nll = nll.mean()
            loss = nll
        else:
            nll = None
            loss = None
        return y_pred_mean, y_pred_var, loss, None, nll  # kl always None
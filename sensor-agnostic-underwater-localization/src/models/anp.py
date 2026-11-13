"""Attentive neural processes and the building blocks shared by all models.

DeterministicModel (attentive CNP) and LatentModel (ANP), each with an optional
spatial front end that encodes the sensor positions. r_anp.py, online_r_anp.py
and pfn.py reuse the layers, encoders, decoders and loss helpers defined here.

Masks are boolean: True for a real point, False for padding.
"""

import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))  # type: ignore[arg-type]

    def forward(self, x):
        return self.linear_layer(x)

def gaussian_nll(mean, var, rho, y):
    """Per-element Gaussian NLL, shape (..., D).

    Without rho the dimensions are independent. With rho, the first two
    dimensions (x, y) form a correlated 2x2 block and the rest stay independent;
    z is left out because it is constant in this data. The joint (x, y) term is
    split evenly across the two slots, so the output shape is unchanged.
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


def gaussian_kl(prior_mu, prior_logvar, post_mu, post_logvar):
    """KL(posterior || prior) between diagonal Gaussians given as (mu, log-variance),
    summed over the latent dims. Returns one value per sample, shape (B,)."""
    kl = (t.exp(post_logvar) + (post_mu - prior_mu) ** 2) / t.exp(prior_logvar) - 1.0 \
         + (prior_logvar - post_logvar)                     # (B, latent_dim)
    return 0.5 * kl.sum(dim=-1)                              # (B,)


def reduce_nll(nll, target_mask=None):
    """Mean of a per-element NLL (B, T, D) over every scored element.

    target_mask: optional bool (B, T), True = score this target point. Without it
    every element counts (a plain mean)."""
    if target_mask is None:
        return nll.mean()
    m = target_mask.unsqueeze(-1)                            # (B, T, 1)
    denom = m.sum().clamp(min=1) * nll.size(-1)
    return (nll * m).sum() / denom


# Decoder heads predict var = var_floor + softplus(.), with the floor in model
# units. With normalized targets the same floor means a different physical floor
# for each target scale; physical_var_floor and set_var_floor set it in physical
# units instead.
DEFAULT_VAR_FLOOR = 1e-3


def physical_var_floor(var_floor_phys, y_std=None, cap=DEFAULT_VAR_FLOOR):
    """Per-dimension floor in model units for a floor given in physical units^2.
    y_std is the target scale when targets are normalized (None otherwise). The
    result is capped at `cap`, so near-constant dimensions keep the default."""
    if y_std is None:
        return t.as_tensor(float(var_floor_phys), dtype=t.float32)
    ys = t.as_tensor(y_std, dtype=t.float32)
    return t.clamp(float(var_floor_phys) / ys ** 2, max=cap)


def set_var_floor(model, floor):
    """Set the variance floor of every decoder head in `model` (each module with a
    var_floor buffer). floor: scalar or per-output-dim tensor, in model units."""
    floor = t.as_tensor(floor, dtype=t.float32)
    heads = [m for m in model.modules() if isinstance(getattr(m, "var_floor", None), t.Tensor)]
    if not heads:
        raise ValueError(f"{type(model).__name__} has no var_floor buffer")
    for m in heads:
        m.var_floor = floor.to(m.var_floor.device)   # stays a registered, non-persistent buffer
    return len(heads)


class LatentEncoder(nn.Module):
    """Set encoder for the latent path: (x, y) points -> q(z) = N(mu, exp(log_var)).

    pooling: how the per-point features are aggregated into one vector.
      'mean'      -- uniform mean over the points (the ANP default).
      'attention' -- learned softmax weights over the points (the RANP default,
                     whose points are RNN states rather than independent samples).
    mask: optional bool (B, N), True = real point. Padded points are excluded
    from the self-attention keys and from the pooling."""

    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1, attn_ffn=False,
                 pooling="mean"):
        super(LatentEncoder, self).__init__()
        assert pooling in ("mean", "attention"), f"pooling must be mean|attention, got {pooling}"
        self.pooling = pooling
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout, ffn=attn_ffn) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_var = Linear(num_hidden, num_latent)
        if pooling == "attention":
            # A plain Linear named pool_attn, for checkpoint compatibility.
            self.pool_attn = nn.Linear(num_hidden, 1)

    def _pool(self, h, mask):
        if self.pooling == "attention":
            scores = self.pool_attn(h)                       # (B, N, 1)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            weights = t.softmax(scores, dim=1)               # (B, N, 1)
            return (h * weights).sum(dim=1)                  # (B, H)
        if mask is None:
            return h.mean(dim=1)
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, x, y, mask=None):
        encoder_input = t.cat([x, y], dim=-1)
        encoder_input = self.input_projection(encoder_input)
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input,
                                         key_padding_mask=mask)

        hidden = self._pool(encoder_input, mask)
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

    def forward(self, context_x, context_y, target_x, context_mask=None):
        # context_mask: optional bool (B, n_ctx), True for real context points.
        encoder_input = t.cat([context_x, context_y], dim=-1)
        encoder_input = self.input_projection(encoder_input)

        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input,
                                         key_padding_mask=context_mask)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query,
                                 key_padding_mask=context_mask)

        return query

class Decoder(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1,
                 full_cov=False):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        # Same LayerNorm + dropout as the deterministic decoder, so the latent
        # and deterministic models differ only in the latent path.
        self.norms = nn.ModuleList([nn.LayerNorm(num_hidden * 3) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.mean_projection = Linear(num_hidden*3, output_dim)
        self.log_var_projection = Linear(num_hidden*3, output_dim)
        # x-y correlation head, created only when full_cov=True.
        self.rho_projection = Linear(num_hidden*3, 1) if full_cov else None
        # Variance floor in model units (see set_var_floor), not saved.
        self.register_buffer("var_floor", t.tensor(DEFAULT_VAR_FLOOR), persistent=False)


    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        target_x = self.target_projection(target_x)
        hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)
        for linear, norm in zip(self.linears, self.norms):
            hidden = t.relu(linear(hidden))
            hidden = norm(hidden)
            hidden = self.dropout(hidden)
        mean = self.mean_projection(hidden)
        var = self.var_floor + F.softplus(self.log_var_projection(hidden))
        rho = (0.99 * t.tanh(self.rho_projection(hidden))
               if self.rho_projection is not None else None)
        return mean, var, rho

class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_k, dropout=0.1):
        super().__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, key, value, query, attn_mask=None):
        """
        key, value: (B', Lk, d)
        query:      (B', Lq, d)
        attn_mask:  optional bool (B', Lq, Lk), True = attend (else masked out).
        """
        # SDPA applies the 1/sqrt(d) scaling internally.
        dropout_p = self.attn_dropout.p if self.training else 0.0

        # scaled_dot_product_attention expects (query, key, value)
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False
        )

        # SDPA does not return attention weights.
        attn_weights = None
        return out, attn_weights


class Attention(nn.Module):
    """Attention sublayer, optionally followed by a feed-forward sublayer.

    With ffn=False the block is attention, concatenation with the residual, a
    linear layer, a residual add and LayerNorm. ffn=True appends a standard
    x + FFN(LayerNorm(x)) sublayer (4x expansion, GELU) after the LayerNorm; its
    parameters are only created when ffn=True.
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

    def forward(self, key, value, query, key_padding_mask=None):
        # key_padding_mask: optional bool (batch_size, seq_k), True = valid key.
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

        # Build the (h*batch, seq_q, seq_k) attend-mask matching the h-major /
        # batch-minor flatten above. Keys are the same across query positions.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (key_padding_mask.view(1, batch_size, 1, seq_k)
                         .expand(self.h, batch_size, seq_q, seq_k)
                         .reshape(self.h * batch_size, seq_q, seq_k))

        result, attns = self.multihead(key, value, query, attn_mask=attn_mask)

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


# =========================================================================== #
# Spatial (sensor-position-aware) front-end encoder
# =========================================================================== #
class FourierPositionEncoding(nn.Module):
    """Sinusoidal encoding of a position at log-spaced wavelengths in meters.
    Positions are encoded in their absolute frame, so the encoding keeps where
    the array is.

    Output dim = in_dim * 2 * n_bands (sin and cos per band and coordinate)."""

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
    """Spatial front end. Turns each point's flat acoustic vector
    (feat_dim = tau * n_sensors) into a num_hidden embedding that is equivariant
    to the order of the sensors and carries each sensor's position.

    Pipeline (tokenize=True): one token per sensor, concatenated with a Fourier
    encoding of its position, a shared projection, self-attention across the
    sensors and pooling.

    Ablation flags:
      tokenize       False keeps the flat vector (optionally with all sensor
                     positions appended) and projects it.
      use_position   add the Fourier position features.
      use_attention  self-attention across sensors (else pooling only).
      pooling        'attention' (learned query) or 'mean'.

    The flat vector is tau-major / sensor-minor, so it is reshaped to
    (..., tau, n_sensors) before the sensor axis is moved.
    """

    def __init__(self, feat_dim, n_sensors, num_hidden, *, tokenize=True,
                 use_position=True, use_attention=True, n_attn_layers=1,
                 pooling="attention", n_fourier_bands=8, min_wavelength=10.0,
                 max_wavelength=1000.0, pos_dim=2, dropout=0.1, norm_acoustic="layernorm",
                 sensor_id_embed=False):
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

        # Raw acoustics are on a much smaller scale than the Fourier features
        # (std ~5e-3 vs ~0.7), so they are rescaled first:
        #   layernorm    per-sensor LayerNorm (loses cross-sensor amplitude ratios)
        #   standardize  one global std computed on the training set
        #   none         no rescaling
        if isinstance(norm_acoustic, bool):
            norm_acoustic = "layernorm" if norm_acoustic else "none"
        self.norm_mode = str(norm_acoustic).lower()
        assert self.norm_mode in ("layernorm", "standardize", "none"), \
            f"norm_acoustic must be layernorm|standardize|none, got {norm_acoustic}"
        norm_dim = self.tau if tokenize else feat_dim
        self.acoustic_norm = nn.LayerNorm(norm_dim) if self.norm_mode == "layernorm" else None
        if self.norm_mode == "standardize":
            # Set from the training data by set_acoustic_stats().
            self.register_buffer("x_mean", t.zeros(1))
            self.register_buffer("x_scale", t.ones(1))

        # Optional learned embedding per sensor slot. Without position features
        # the tokens are exchangeable and nothing identifies a sensor across
        # points; the slot embedding restores that. Off by default.
        self.sensor_id_embed = bool(sensor_id_embed) and tokenize
        if self.sensor_id_embed:
            self.sensor_emb = nn.Parameter(t.randn(1, n_sensors, num_hidden) * 0.02)

        if tokenize:
            # per-sensor token = (tau acoustic) [+ position embedding], shared proj
            self.token_proj = Linear(self.tau + pos_out, num_hidden, w_init='relu')
            if self.use_attention:
                self.attn_layers = nn.ModuleList(
                    [Attention(num_hidden, dropout=dropout) for _ in range(n_attn_layers)])
            if pooling == "attention":
                self.pool_query = nn.Parameter(t.randn(1, 1, num_hidden) * 0.02)
                self.pool_attn = Attention(num_hidden, dropout=dropout)
        else:
            # flat control: raw feat vector [+ all sensors' position embeddings]
            self.flat_proj = Linear(feat_dim + n_sensors * pos_out, num_hidden,
                                    w_init='relu')

    def set_acoustic_stats(self, mean, std):
        """Set global acoustic mean/std for the 'standardize' mode (no-op otherwise).
        Called by the trainer after computing stats on the train set."""
        if self.norm_mode == "standardize":
            self.x_mean.fill_(float(mean))
            self.x_scale.fill_(float(std) + 1e-8)

    def _norm_acoustic(self, a):
        if self.norm_mode == "layernorm":
            return self.acoustic_norm(a)
        if self.norm_mode == "standardize":
            return (a - self.x_mean) / self.x_scale
        return a

    def forward(self, x, sensor_pos, sensor_mask=None):
        # x: (B, N, feat_dim); sensor_pos: (B, n_sensors, 3), the same for all
        # N points. Returns (B, N, num_hidden). sensor_mask: optional bool
        # (B, n_sensors), True for present sensors; missing sensors are left
        # out of the attention and the pooling (tokenized path only).
        B, N, F = x.shape
        pos = sensor_pos[..., :self.pos_dim] if self.use_position else None  # (B,S,pd)

        if not self.tokenize:
            if sensor_mask is not None:
                raise ValueError("sensor_mask is only supported for the tokenized "
                                 "SpatialEncoder (tokenize=True), not the flat control")
            xa = self._norm_acoustic(x)
            if self.use_position:
                pe = self.pos_enc(pos).reshape(B, -1)          # (B, S*pos_out)
                pe = pe[:, None, :].expand(B, N, -1)           # broadcast over points
                xa = t.cat([xa, pe], dim=-1)
            return t.relu(self.flat_proj(xa))

        # tau-major / sensor-minor -> (B, N, tau, S) -> (B, N, S, tau)
        tok = x.view(B, N, self.tau, self.n_sensors).transpose(-1, -2)
        tok = self._norm_acoustic(tok)                        # balance acoustic vs position scale
        if self.use_position:
            pe = self.pos_enc(pos)                             # (B, S, pos_out)
            pe = pe[:, None, :, :].expand(B, N, self.n_sensors, -1)
            tok = t.cat([tok, pe], dim=-1)                     # (B, N, S, tau+pos_out)
        tok = self.token_proj(tok)                            # (B, N, S, H)

        if self.sensor_id_embed:
            tok = tok + self.sensor_emb[:, None, :, :]      # (1,1,S,H) broadcast
        tok = tok.reshape(B * N, self.n_sensors, self.num_hidden)
        # (B*N, S) key-padding mask shared across the N points of each trajectory.
        smask = (sensor_mask[:, None, :].expand(B, N, self.n_sensors).reshape(B * N, self.n_sensors)
                 if sensor_mask is not None else None)
        if self.use_attention:
            for attn in self.attn_layers:
                tok, _ = attn(tok, tok, tok, key_padding_mask=smask)   # self-attn over sensors
        if self.pooling == "attention":
            q = self.pool_query.expand(B * N, -1, -1)          # (B*N, 1, H)
            pooled, _ = self.pool_attn(tok, tok, q, key_padding_mask=smask)  # query attends sensors
            pooled = pooled.squeeze(1)
        elif smask is not None:
            m = smask[..., None].to(tok.dtype)                 # masked Deep-Sets mean
            pooled = (tok * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = tok.mean(dim=1)                           # Deep-Sets mean pool
        return pooled.view(B, N, self.num_hidden)


class SensorSetEncoder(nn.Module):
    """Alternative sensor-set encoder (model.spatial.encoder=set), with the same
    interface as SpatialEncoder. Sensor positions are taken relative to the
    array centroid."""

    def __init__(self, feat_dim, n_sensors, num_hidden, *, d_model=256, n_layers=6,
                 n_heads=8, pos_scale=100.0, dropout=0.0, bf16=True):
        super().__init__()
        assert feat_dim % n_sensors == 0, \
            f"feat_dim {feat_dim} not divisible by n_sensors {n_sensors}"
        self.n_sensors = n_sensors
        self.tau = feat_dim // n_sensors
        self.num_hidden = num_hidden
        self.d_model = d_model
        self.pos_scale = float(pos_scale)
        self.bf16 = bool(bf16)
        self.norm_mode = "standardize"                  # the trainer sets the stats
        self.register_buffer("x_mean", t.zeros(1))
        self.register_buffer("x_scale", t.ones(1))
        self.inp = nn.Sequential(nn.Linear(self.tau + 2, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        self.readout = nn.Parameter(t.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_hidden))

    def set_acoustic_stats(self, mean, std):
        self.x_mean.fill_(float(mean))
        self.x_scale.fill_(float(std) + 1e-8)

    def forward(self, x, sensor_pos, sensor_mask=None):
        B, N, _ = x.shape
        S, d = self.n_sensors, self.d_model
        tok = x.view(B, N, self.tau, S).transpose(-1, -2)            # (B, N, S, tau)
        tok = (tok - self.x_mean) / self.x_scale
        pos = sensor_pos[..., :2]
        rel = (pos - pos.mean(1, keepdim=True)) / self.pos_scale     # (B, S, 2)
        rel = rel[:, None].expand(B, N, S, 2)
        kpm = None
        if sensor_mask is not None:                                   # True = padded key
            kpm = ~sensor_mask[:, None, :].expand(B, N, S).reshape(B * N, S)
            kpm = t.cat([t.zeros(B * N, 1, dtype=t.bool, device=x.device), kpm], 1)
        with t.autocast("cuda", dtype=t.bfloat16, enabled=self.bf16 and x.is_cuda):
            h = self.inp(t.cat([tok, rel.to(tok.dtype)], -1)).reshape(B * N, S, d)
            h = t.cat([self.readout.expand(B * N, -1, -1), h], 1)
            h = self.layers(h, src_key_padding_mask=kpm)
            h = self.out(h[:, 0])
        return h.float().view(B, N, self.num_hidden)


def build_spatial_encoder(spatial_cfg, feat_dim, num_hidden, dropout):
    """Instantiate a SpatialEncoder from a config mapping, or None if disabled.
    Returns (encoder_or_None, encoder_output_dim). When enabled the downstream NP
    encoders take ``num_hidden`` as their input_dim instead of the raw feat_dim."""
    if not spatial_cfg:
        return None, feat_dim
    get = spatial_cfg.get
    if not bool(get("enabled", False)):
        return None, feat_dim
    kind = str(get("encoder", "legacy")).lower()
    if kind == "set":
        enc = SensorSetEncoder(
            feat_dim=feat_dim, n_sensors=int(get("n_sensors", 10)), num_hidden=num_hidden,
            d_model=int(get("set_d_model", 256)), n_layers=int(get("set_layers", 6)),
            n_heads=int(get("set_heads", 8)), pos_scale=float(get("set_pos_scale", 100.0)),
            dropout=float(get("set_dropout", 0.0)), bf16=bool(get("set_bf16", True)))
        return enc, num_hidden
    if kind != "legacy":
        raise ValueError(f"model.spatial.encoder must be legacy|set, got {kind}")
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
        norm_acoustic=get("norm_acoustic", "layernorm"), dropout=dropout,
        sensor_id_embed=bool(get("sensor_id_embed", False)))
    return enc, num_hidden


# LatentModel:

class LatentModel(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1, spatial_cfg=None, full_cov=False, attn_ffn=False,
                 latent_pooling="mean"):
        super(LatentModel, self).__init__()
        # Optional spatial front end: (B, N, feat_dim) -> (B, N, num_hidden).
        self.spatial_encoder, enc_input_dim = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.latent_encoder = LatentEncoder(num_hidden, num_latent=num_hidden,
                                            input_dim=enc_input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout,
                                            attn_ffn=attn_ffn,
                                            pooling=latent_pooling)
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=enc_input_dim,
                                                          output_dim=output_dim,
                                                          dropout=dropout,
                                                          attn_ffn=attn_ffn)
        self.decoder = Decoder(num_hidden,
                               input_dim=enc_input_dim,
                               output_dim=output_dim,
                               dropout=dropout, full_cov=full_cov)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0,
                predict_with_prior: bool = False, sensor_pos=None, sensor_mask=None,
                context_mask=None, target_mask=None):
        # sensor_mask: optional bool (B, n_sensors), True for present sensors.
        # predict_with_prior: decode from the prior latent (context only) even
        #   when target_y is given; the posterior and the KL are still computed.
        #   Used for validation; training decodes from the posterior.
        # context_mask: optional bool (B, n_ctx), True for real context points.
        # target_mask: optional bool (B, T), True for scored targets; unscored
        #   targets are also left out of the posterior.
        if self.spatial_encoder is not None:
            # sensor_pos is per-trajectory (same geometry for context & targets).
            context_x = self.spatial_encoder(context_x, sensor_pos, sensor_mask=sensor_mask)
            target_x = self.spatial_encoder(target_x, sensor_pos, sensor_mask=sensor_mask)
        num_targets = target_x.size(1)
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y, mask=context_mask)

        posterior_mu = posterior_var = None
        if target_y is not None:
            # The posterior conditions on context and targets (training only).
            post_mask = None
            if context_mask is not None or target_mask is not None:
                B = context_x.size(0)
                cm = (context_mask if context_mask is not None else
                      t.ones(B, context_x.size(1), dtype=t.bool, device=context_x.device))
                tm = (target_mask if target_mask is not None else
                      t.ones(B, num_targets, dtype=t.bool, device=target_x.device))
                post_mask = t.cat([cm, tm], dim=1)
            posterior_mu, posterior_var, posterior = self.latent_encoder(
                t.cat([context_x, target_x], dim=1),
                t.cat([context_y, target_y], dim=1),
                mask=post_mask)

        use_posterior = (target_y is not None) and (not predict_with_prior)
        # Sample z in training; use the mean of the distribution in eval mode.
        if use_posterior:
            z = posterior if self.training else posterior_mu
        else:
            z = prior if self.training else prior_mu

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(context_x, context_y, target_x, context_mask=context_mask)

        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            nll = reduce_nll(nll, target_mask)
            # The NLL is a mean over targets and output dims while the KL sums
            # over latent dims, so the KL is normalized to the same per-point
            # units (Le et al., 2018); beta=1 then gives the standard ELBO.
            kl_per_sample = gaussian_kl(prior_mu, prior_var, posterior_mu, posterior_var)
            if target_mask is None:
                kl = kl_per_sample.mean() / (num_targets * target_y.size(-1))
            else:
                # Per-sample target counts: divide by the number of scored
                # elements, as reduce_nll does.
                kl = kl_per_sample.sum() / (target_mask.sum().clamp(min=1) * target_y.size(-1))
            loss = nll + beta * kl
        else:
            kl = None
            loss = None
            nll = None
        return y_pred_mean, y_pred_var, loss, kl, nll

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        # prior_var / posterior_var are log-variances; batch-mean KL (scalar).
        return gaussian_kl(prior_mu, prior_var, posterior_mu, posterior_var).mean()


class DeterministicDecoder(nn.Module):
    """Decoder for CNP: takes only the deterministic representation r (no latent z)."""
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
        self.register_buffer("var_floor", t.tensor(DEFAULT_VAR_FLOOR), persistent=False)

    def forward(self, r, target_x):
        target_x = self.target_projection(target_x)
        hidden = t.cat([r, target_x], dim=-1)
        for linear, norm in zip(self.linears, self.norms):
            hidden = t.relu(linear(hidden))
            hidden = norm(hidden)
            hidden = self.dropout(hidden)
        mean = self.mean_projection(hidden)
        var = self.var_floor + F.softplus(self.log_var_projection(hidden))
        rho = (0.99 * t.tanh(self.rho_projection(hidden))
               if self.rho_projection is not None else None)
        return mean, var, rho


class DeterministicModel(nn.Module):
    """CNP: attentive deterministic encoder + decoder, no latent variable."""
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1, spatial_cfg=None, full_cov=False, attn_ffn=False):
        super(DeterministicModel, self).__init__()
        self.spatial_encoder, enc_input_dim = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=enc_input_dim,
                                                          output_dim=output_dim,
                                                          dropout=dropout,
                                                          attn_ffn=attn_ffn)
        self.decoder = DeterministicDecoder(num_hidden,
                                            input_dim=enc_input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout, full_cov=full_cov)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0,
                predict_with_prior: bool = False, sensor_pos=None, context_mask=None,
                target_mask=None, sensor_mask=None):
        # predict_with_prior has no effect here; it keeps the interface of the
        # latent models. The masks are as in LatentModel.forward.
        if self.spatial_encoder is not None:
            context_x = self.spatial_encoder(context_x, sensor_pos, sensor_mask=sensor_mask)
            target_x = self.spatial_encoder(target_x, sensor_pos, sensor_mask=sensor_mask)
        r = self.deterministic_encoder(context_x, context_y, target_x, context_mask=context_mask)
        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            nll = reduce_nll(nll, target_mask)
            loss = nll
        else:
            nll = None
            loss = None
        return y_pred_mean, y_pred_var, loss, None, nll  # kl always None
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
    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1, attn_ffn=False):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout, ffn=attn_ffn) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout, ffn=attn_ffn) for _ in range(2)])
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.context_projection = Linear(input_dim, num_hidden)
        self.target_projection = Linear(input_dim, num_hidden)

    def forward(self, context_x, context_y, target_x, context_mask=None):
        # context_mask: optional bool (B, n_ctx), True = real context point (else
        # a padded slot to be ignored). Lets a batch hold per-sample context sizes.
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
        # Per-layer LayerNorm + dropout, matching the CNP's DeterministicDecoder
        # (and the RANP decoder in r_anp.py) so ANP-vs-CNP comparisons differ in
        # the latent path only, not in decoder regularization.
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

    def forward(self, key, value, query, attn_mask=None):
        """
        key, value: (B', Lk, d)
        query:      (B', Lq, d)
        attn_mask:  optional bool (B', Lq, Lk), True = attend (else masked out).
        """
        # SDPA apply scale 1/sqrt(d) internally and can use optimized kernels
        dropout_p = self.attn_dropout.p if self.training else 0.0

        # scaled_dot_product_attention expects (query, key, value)
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False
        )

        # For max perf, SDPA does not return attention weights.
        attn_weights = None
        return out, attn_weights


class Attention(nn.Module):
    """Attention sublayer, optionally followed by a feed-forward sublayer.

    ``ffn=False`` (default) is the ORIGINAL block, bit-identical: attention,
    concat-with-residual, linear, residual add, LayerNorm -- and no feed-forward
    stage at all. That omission is measurable: on the 8000-layout layout-OOD task
    this block reaches val 12.47 / train 21.88 where a standard pre-LN transformer
    block of the same width reaches 8.28 / 11.02, and a capacity-matched
    comparison attributed ~74% of that gap to architecture rather than size. The
    missing piece is the per-token nonlinearity below.

    ``ffn=True`` appends the standard sublayer -- ``x + FFN(LayerNorm(x))`` with a
    4x inner expansion and GELU -- leaving the attention path untouched. It is
    added AFTER the existing LayerNorm as a residual branch, so with the new
    weights at their initialisation the block still computes very nearly the
    original function and training starts from a comparable point.

    Gated off by default so every existing config, checkpoint and result is
    unaffected: when ``ffn=False`` no parameters are created and the state_dict
    keys are unchanged.
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

        # CRITICAL: raw acoustics have std ~5e-3 vs ~0.7 for the Fourier position
        # features, so without balancing them the position swamps the token and
        # the model collapses to constant prediction. Modes:
        #   layernorm   -- per-sensor LayerNorm; unit-scales each spectrum but
        #                  DESTROYS cross-sensor amplitude ratios (a range cue).
        #   standardize -- divide by ONE global std (set from train data); keeps
        #                  the relative amplitudes across sensors/bins/points.
        #   none        -- no balancing (collapses; ablation only).
        if isinstance(norm_acoustic, bool):
            norm_acoustic = "layernorm" if norm_acoustic else "none"
        self.norm_mode = str(norm_acoustic).lower()
        assert self.norm_mode in ("layernorm", "standardize", "none"), \
            f"norm_acoustic must be layernorm|standardize|none, got {norm_acoustic}"
        norm_dim = self.tau if tokenize else feat_dim
        self.acoustic_norm = nn.LayerNorm(norm_dim) if self.norm_mode == "layernorm" else None
        if self.norm_mode == "standardize":
            # global scalar stats; populated from data via set_acoustic_stats().
            self.register_buffer("x_mean", t.zeros(1))
            self.register_buffer("x_scale", t.ones(1))

        # Learned per-SENSOR-SLOT embedding. The shared token_proj + permutation-
        # invariant pool make the sensors exchangeable, which is the right bias
        # when each token already carries its POSITION. With use_position=False
        # it is fatal: the point embedding becomes an unordered multiset of the
        # per-sensor measurements, so nothing links "sensor 3" across context
        # points -- and multilaterating a layout from a calibration set requires
        # exactly that correspondence. Sensor slot IS a stable label within a
        # deployment, so tagging it restores the correspondence while keeping
        # the cross-sensor attention. Default False = original behaviour.
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
        # x: (B, N, feat_dim);  sensor_pos: (B, n_sensors, 3), constant over the
        # N points of a trajectory (same geometry). Returns (B, N, num_hidden).
        # sensor_mask: optional bool (B, n_sensors), True = present sensor. A
        # False sensor is dropped from the cross-sensor attention and the pool,
        # so the point embedding is built only from surviving sensors -> lets us
        # evaluate graceful degradation as sensors fail (tokenized path only).
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
        norm_acoustic=get("norm_acoustic", "layernorm"), dropout=dropout,
        sensor_id_embed=bool(get("sensor_id_embed", False)))
    return enc, num_hidden


# LatentModel:

class LatentModel(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1, spatial_cfg=None, full_cov=False, attn_ffn=False):
        super(LatentModel, self).__init__()
        # Optional sensor-position-aware front end. When enabled it maps the raw
        # (B, N, feat_dim) acoustics to (B, N, num_hidden), so the NP encoders see
        # num_hidden-dim inputs instead of the flat feat_dim.
        self.spatial_encoder, enc_input_dim = build_spatial_encoder(
            spatial_cfg, input_dim, num_hidden, dropout)
        self.latent_encoder = LatentEncoder(num_hidden, num_latent=num_hidden,
                                            input_dim=enc_input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout,
                                            attn_ffn=attn_ffn)
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
                predict_with_prior: bool = False, sensor_pos=None, sensor_mask=None):
        # sensor_mask: optional bool (B, n_sensors), True = present; drops failed
        # sensors from the spatial front end (eval-time robustness study).
        # predict_with_prior: if True the DECODER is driven by the PRIOR latent
        # (context only) even when target_y is supplied, the deployment-faithful
        # path (at inference the latent cannot peek at target labels). The
        # posterior + KL/NLL are still computed when target_y is given, so the
        # validation loss stays comparable to training; only the z that drives the
        # prediction changes. Use False for training (posterior teacher forcing).
        if self.spatial_encoder is not None:
            # sensor_pos is per-trajectory (same geometry for context & targets).
            context_x = self.spatial_encoder(context_x, sensor_pos, sensor_mask=sensor_mask)
            target_x = self.spatial_encoder(target_x, sensor_pos, sensor_mask=sensor_mask)
        num_targets = target_x.size(1)
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        posterior_mu = posterior_var = None
        if target_y is not None:
            # Posterior conditions on context UNION targets (standard ANP; also
            # what online_r_anp.py does): the best-informed distribution the KL
            # teaches the context-only prior to approximate. Training-only
            # machinery -- deployment always runs the prior path (target_y=None).
            posterior_mu, posterior_var, posterior = self.latent_encoder(
                t.cat([context_x, target_x], dim=1),
                t.cat([context_y, target_y], dim=1))

        use_posterior = (target_y is not None) and (not predict_with_prior)
        # train(): decode a SAMPLE of z (stochastic ELBO). eval(): decode the
        # distribution MEAN -- the point-prediction analogue of disabling dropout;
        # a single sampled z adds zero-mean noise that only inflates MAE and makes
        # checkpoint selection jittery. Sampling at eval is still available by
        # calling with the module in train() mode.
        if use_posterior:
            z = posterior if self.training else posterior_mu
        else:
            z = prior if self.training else prior_mu

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(context_x, context_y, target_x)

        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            nll = nll.mean()
            # NLL is a MEAN over batch x targets x output dims while kl_div SUMS
            # over the latent dims, so an unnormalized `nll + beta*kl` weighs the
            # KL ~ num_targets*output_dim times more than the per-point ELBO
            # (~120x here) and crushes the posterior onto the prior. Normalize the
            # KL to the same per-target-point, per-dim units (Le et al. 2018) so
            # beta=1 reads as the standard ELBO.
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var) #type: ignore[assignment]
            kl = kl / (num_targets * target_y.size(-1))
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
        # predict_with_prior is accepted for a uniform interface with the latent
        # models but has no effect: a deterministic CNP never uses target labels
        # to predict, so there is no posterior to peek at.
        # context_mask: optional bool (B, n_ctx) marking real vs padded context
        # slots, so one batch can carry per-sample context sizes (masked pooling).
        # target_mask: optional bool (B, T) marking which target points to SCORE
        # (per-sample; e.g. exclude each sample's own context points). Both come
        # from the per-sample context-batching path (data.context.per_sample).
        # sensor_mask: optional bool (B, n_sensors), True = present (sensor-drop study).
        if self.spatial_encoder is not None:
            context_x = self.spatial_encoder(context_x, sensor_pos, sensor_mask=sensor_mask)
            target_x = self.spatial_encoder(target_x, sensor_pos, sensor_mask=sensor_mask)
        r = self.deterministic_encoder(context_x, context_y, target_x, context_mask=context_mask)
        y_pred_mean, y_pred_var, y_pred_rho = self.decoder(r, target_x)

        if target_y is not None:
            nll = gaussian_nll(y_pred_mean, y_pred_var, y_pred_rho, target_y)
            if target_mask is not None:
                m = target_mask.unsqueeze(-1)                       # (B, T, 1)
                denom = m.sum().clamp(min=1) * y_pred_mean.size(-1)
                nll = (nll * m).sum() / denom
            else:
                nll = nll.mean()
            loss = nll
        else:
            nll = None
            loss = None
        return y_pred_mean, y_pred_var, loss, None, nll  # kl always None
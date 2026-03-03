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
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim, output_dim):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
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
    def __init__(self, num_hidden, num_latent, input_dim, output_dim):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
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
    def __init__(self, num_hidden, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
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
    def __init__(self, num_hidden_k):
        super().__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

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
    def __init__(self, num_hidden, h=4):
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.residual_dropout = nn.Dropout(p=0.1)
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

    def forward(self, x_seq: t.Tensor) -> t.Tensor:
        # x_seq: (B, T, Dx)
        h_seq, _ = self.rnn(x_seq)      # (B, T, Dh)
        h_seq = self.norm(h_seq)
        return h_seq

# LatentModel:

class LatentModel(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_latent=num_hidden,
                                            input_dim=input_dim,
                                            output_dim=output_dim)
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=input_dim,
                                                          output_dim=output_dim)
        self.decoder = Decoder(num_hidden,
                               input_dim=input_dim,
                               output_dim=output_dim)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0):
        num_targets = target_x.size(1)
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        r = self.deterministic_encoder(context_x, context_y, target_x)

        y_pred_mean, y_pred_var = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = 0.5 * t.log(2 * t.pi * y_pred_var) + 0.5 * ((target_y - y_pred_mean) ** 2) / y_pred_var
            nll = nll.mean()
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
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


# =========================================================================
# Sequential (Recurrent) ANP  —  SNP with per-step latent updates
# =========================================================================

class SequentialLatentEncoder(nn.Module):
    """
    RNN-based latent encoder for Sequential ANP.
    Processes (x_t, y_t) pairs step-by-step via an LSTM/GRU.
    Returns per-step latent distribution parameters (mu_t, log_var_t).
    """
    def __init__(self, input_dim, output_dim, num_hidden, num_latent,
                 rnn_type="lstm", num_layers=1, dropout=0.0):
        super().__init__()
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=num_hidden,
            hidden_size=num_hidden,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(num_hidden)
        self.penultimate = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_var = Linear(num_hidden, num_latent)

    def forward(self, x_seq, y_seq):
        """
        x_seq: (B, T, Dx)
        y_seq: (B, T, Dy)  — shifted y for prior, actual y for posterior
        Returns: mu (B, T, latent), log_var (B, T, latent)
        """
        inp = t.cat([x_seq, y_seq], dim=-1)   # (B, T, Dx+Dy)
        inp = self.input_projection(inp)       # (B, T, H)
        h_seq, _ = self.rnn(inp)               # (B, T, H)
        h_seq = self.norm(h_seq)
        h_seq = t.relu(self.penultimate(h_seq))
        mu = self.mu(h_seq)                    # (B, T, latent)
        log_var = self.log_var(h_seq)          # (B, T, latent)
        log_var = 3 * t.tanh(log_var)          # bound log-variance
        return mu, log_var


class SequentialDeterministicEncoder(nn.Module):
    """
    RNN-based deterministic encoder for Sequential ANP.
    Processes (x_t, y_t) pairs sequentially to produce per-step
    deterministic representations r_t.
    """
    def __init__(self, input_dim, output_dim, num_hidden,
                 rnn_type="lstm", num_layers=1, dropout=0.0):
        super().__init__()
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=num_hidden,
            hidden_size=num_hidden,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(num_hidden)

    def forward(self, x_seq, y_seq):
        """
        x_seq: (B, T, Dx)
        y_seq: (B, T, Dy)
        Returns: r_seq (B, T, H)
        """
        inp = t.cat([x_seq, y_seq], dim=-1)
        inp = self.input_projection(inp)
        r_seq, _ = self.rnn(inp)
        r_seq = self.norm(r_seq)
        return r_seq


class SequentialLatentModel(nn.Module):
    """
    Sequential (Recurrent) ANP that processes time series step-by-step.

    Prior path:     conditions on (x_t, y_{t-1})  — shifted y, no peek at current y_t
    Posterior path:  conditions on (x_t, y_t)      — sees current y_t (training only)
    Deterministic:   conditions on (x_t, y_{t-1})  — same as prior

    At each step t the model forms:
        prior     p(z_t | x_{1:t}, y_{1:t-1})
        posterior  q(z_t | x_{1:t}, y_{1:t})
    and decodes  ŷ_t = Decoder(r_t, z_t, x_t).

    Loss = mean_t[NLL_t] + β · mean_t[KL_t]
    """
    def __init__(self, num_hidden, input_dim, output_dim,
                 rnn_type="lstm", rnn_layers=1, rnn_dropout=0.0):
        super().__init__()
        self.num_hidden = num_hidden
        self.output_dim = output_dim

        self.latent_encoder = SequentialLatentEncoder(
            input_dim=input_dim, output_dim=output_dim,
            num_hidden=num_hidden, num_latent=num_hidden,
            rnn_type=rnn_type, num_layers=rnn_layers, dropout=rnn_dropout,
        )
        self.det_encoder = SequentialDeterministicEncoder(
            input_dim=input_dim, output_dim=output_dim,
            num_hidden=num_hidden,
            rnn_type=rnn_type, num_layers=rnn_layers, dropout=rnn_dropout,
        )
        self.decoder = Decoder(num_hidden, input_dim=input_dim, output_dim=output_dim)

    @staticmethod
    def _shift_y(y_seq):
        """Shift y right by 1: y_shifted[:,0]=0, y_shifted[:,t]=y[:,t-1]."""
        y_shifted = t.zeros_like(y_seq)
        y_shifted[:, 1:, :] = y_seq[:, :-1, :]
        return y_shifted

    def forward(self, x_seq, y_seq, target_y=None, beta: float = 1.0):
        """
        Sequential ANP forward pass with teacher forcing.

        Args:
            x_seq:    (B, T, Dx)  input features (already masked / augmented)
            y_seq:    (B, T, Dy)  ground-truth y used to build the shifted prior
                                  input AND (when target_y is given) the posterior.
            target_y: (B, T, Dy)  if provided → training mode: compute posterior & loss.
                                  if None    → inference mode: sample from prior only.
            beta:     KL weight

        Returns:
            y_pred_mean  (B, T, Dy)
            y_pred_var   (B, T, Dy)
            loss, kl, nll  (scalars or None)
        """
        y_shifted = self._shift_y(y_seq)                          # (B, T, Dy)

        # Prior: conditions on (x_t, y_{t-1})
        prior_mu, prior_log_var = self.latent_encoder(x_seq, y_shifted)

        # Deterministic path: same prior conditioning
        r_seq = self.det_encoder(x_seq, y_shifted)                # (B, T, H)

        if target_y is not None:
            # Training: compute posterior and sample z from it
            post_mu, post_log_var = self.latent_encoder(x_seq, target_y)
            std = t.exp(0.5 * post_log_var)
            std = t.clamp(std, min=1e-6, max=1e6)
            eps = t.randn_like(std)
            z = eps * std + post_mu                               # (B, T, latent)
        else:
            # Inference: sample z from prior
            std = t.exp(0.5 * prior_log_var)
            std = t.clamp(std, min=1e-6, max=1e6)
            eps = t.randn_like(std)
            z = eps * std + prior_mu

        # Decode all steps at once (teacher-forced)
        y_pred_mean, y_pred_var = self.decoder(r_seq, z, x_seq)  # (B, T, Dy)

        if target_y is not None:
            # Per-step NLL, averaged over B, T, Dy
            nll = 0.5 * t.log(2 * math.pi * y_pred_var) \
                  + 0.5 * ((target_y - y_pred_mean) ** 2) / y_pred_var
            nll = nll.mean()

            # Per-step KL, summed over latent dim, averaged over B and T
            kl = self.kl_div(prior_mu, prior_log_var, post_mu, post_log_var)

            loss = nll + beta * kl
        else:
            nll = None
            kl = None
            loss = None

        return y_pred_mean, y_pred_var, loss, kl, nll

    def kl_div(self, prior_mu, prior_log_var, post_mu, post_log_var):
        """KL(q || p) for diagonal Gaussians, per step, averaged over B & T."""
        kl = (t.exp(post_log_var) + (post_mu - prior_mu) ** 2) / t.exp(prior_log_var) \
             - 1.0 + (prior_log_var - post_log_var)              # (B, T, latent)
        kl = 0.5 * kl.sum(dim=-1)                                # (B, T)
        return kl.mean()                                          # scalar

    # -----------------------------------------------------------------
    # Autoregressive inference (no teacher forcing beyond context)
    # -----------------------------------------------------------------

    @t.no_grad()
    def infer_autoregressive(self, x_seq, y_context_norm, n_context):
        """
        Proper test-time inference for Sequential ANP.

        Phase 1 (context):  steps 0 … n_context-1
            • Shifted GT y is fed to the encoders (same as training).
            • RNN hidden states are accumulated.

        Phase 2 (autoregressive):  steps n_context … T-1
            • Each step receives the model's OWN predicted ŷ_{t-1}
              instead of the ground-truth y_{t-1}.
            • RNN hidden states carry over from the context phase.

        All inputs / outputs are in **normalised** y-space.

        Args:
            x_seq:          (B, T, Dx)  full input sequence
            y_context_norm: (B, T, Dy)  normalised GT y — only the first
                            n_context columns are used; the rest are ignored.
            n_context:      int, number of context (observed) steps

        Returns:
            y_pred_mean  (B, T, Dy)   predicted means  (normalised space)
            y_pred_var   (B, T, Dy)   predicted variances (normalised space)
        """
        B, T, Dx = x_seq.shape
        Dy = self.output_dim
        device = x_seq.device

        # ── Phase 1: context chunk ────────────────────────────────────
        # Build shifted y for context → [0, y_0, y_1, …, y_{C-2}]
        y_shifted_ctx = t.zeros(B, n_context, Dy, device=device)
        if n_context > 1:
            y_shifted_ctx[:, 1:, :] = y_context_norm[:, :n_context - 1, :]

        x_ctx = x_seq[:, :n_context, :]

        # Latent encoder — context
        lat_inp = t.cat([x_ctx, y_shifted_ctx], dim=-1)
        lat_inp = self.latent_encoder.input_projection(lat_inp)
        lat_h, lat_hidden = self.latent_encoder.rnn(lat_inp)
        lat_h = self.latent_encoder.norm(lat_h)
        lat_hp = t.relu(self.latent_encoder.penultimate(lat_h))
        mu_ctx = self.latent_encoder.mu(lat_hp)
        lv_ctx = 3 * t.tanh(self.latent_encoder.log_var(lat_hp))
        std_ctx = t.exp(0.5 * lv_ctx).clamp(1e-6, 1e6)
        z_ctx = mu_ctx + std_ctx * t.randn_like(std_ctx)         # (B, C, latent)

        # Deterministic encoder — context
        det_inp = t.cat([x_ctx, y_shifted_ctx], dim=-1)
        det_inp = self.det_encoder.input_projection(det_inp)
        det_h, det_hidden = self.det_encoder.rnn(det_inp)
        r_ctx = self.det_encoder.norm(det_h)                     # (B, C, H)

        # Decode context steps (batch)
        mean_ctx, var_ctx = self.decoder(r_ctx, z_ctx, x_ctx)    # (B, C, Dy)

        if n_context >= T:
            return mean_ctx, var_ctx

        # ── Phase 2: autoregressive rollout ───────────────────────────
        all_means = [mean_ctx]
        all_vars  = [var_ctx]

        # First y_prev is the LAST GT context value (normalised)
        y_prev = y_context_norm[:, n_context - 1:n_context, :]   # (B, 1, Dy)

        for step in range(n_context, T):
            x_t = x_seq[:, step:step + 1, :]                     # (B, 1, Dx)

            # Latent encoder — single step
            inp_lat = t.cat([x_t, y_prev], dim=-1)
            inp_lat = self.latent_encoder.input_projection(inp_lat)
            h_lat, lat_hidden = self.latent_encoder.rnn(inp_lat, lat_hidden)
            h_lat = self.latent_encoder.norm(h_lat)
            h_lat = t.relu(self.latent_encoder.penultimate(h_lat))
            mu_t = self.latent_encoder.mu(h_lat)
            lv_t = 3 * t.tanh(self.latent_encoder.log_var(h_lat))
            std_t = t.exp(0.5 * lv_t).clamp(1e-6, 1e6)
            z_t = mu_t + std_t * t.randn_like(std_t)

            # Deterministic encoder — single step
            inp_det = t.cat([x_t, y_prev], dim=-1)
            inp_det = self.det_encoder.input_projection(inp_det)
            h_det, det_hidden = self.det_encoder.rnn(inp_det, det_hidden)
            r_t = self.det_encoder.norm(h_det)

            # Decode
            mean_t, var_t = self.decoder(r_t, z_t, x_t)          # (B, 1, Dy)
            all_means.append(mean_t)
            all_vars.append(var_t)

            # Feed own prediction as next y_prev (autoregressive)
            y_prev = mean_t                                       # (B, 1, Dy)

        return t.cat(all_means, dim=1), t.cat(all_vars, dim=1)
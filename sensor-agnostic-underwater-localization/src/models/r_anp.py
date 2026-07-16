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
class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim, output_dim, dropout=0.1):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden, dropout=dropout) for _ in range(2)])
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
        # Per-layer LayerNorm + dropout in the decoder MLP.
        self.norms = nn.ModuleList([nn.LayerNorm(num_hidden * 3) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.mean_projection = Linear(num_hidden*3, output_dim)
        self.log_var_projection = Linear(num_hidden*3, output_dim)


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
        ):
        super(LatentModel, self).__init__()
        # RNN integrado: input_dim → num_hidden
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
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
        )
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_latent=num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
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
        ):
        # predict_with_prior: if True the DECODER is driven by the PRIOR latent
        # (context only) even when target_y is supplied, deployment-faithful
        # inference. Posterior + KL/NLL are still computed for logging; only the z
        # driving the prediction changes. False = training (teacher forcing).
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
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)

        use_posterior = (target_y is not None) and (not predict_with_prior)
        z = posterior if use_posterior else prior

        z = z.unsqueeze(1).repeat(1, num_targets, 1)

        # Camino determinista (cross-attention)
        r = self.deterministic_encoder(context_x, context_y, target_x)

        # Decoder
        y_pred_mean, y_pred_var = self.decoder(r, z, target_x)

        if target_y is not None:
            nll = 0.5 * t.log(2 * t.pi * y_pred_var) + \
                  0.5 * ((target_y - y_pred_mean) ** 2) / y_pred_var
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
    ):
        super(DeterministicModel, self).__init__()
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
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
        )
        self.decoder = DeterministicDecoder(
            num_hidden,
            input_dim=num_hidden,
            output_dim=output_dim,
            dropout=dropout,
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
    ):
        # predict_with_prior accepted for interface parity with the latent RANP;
        # a deterministic RCNP never peeks at target labels, so it has no effect.
        h_seq = self.temporal_encoder(x_seq)
        context_x = h_seq[:, context_indices, :]
        target_x  = h_seq[:, target_indices,  :]

        r = self.deterministic_encoder(context_x, context_y, target_x)
        y_pred_mean, y_pred_var = self.decoder(r, target_x)

        if target_y is not None:
            nll = 0.5 * t.log(2 * t.pi * y_pred_var) + \
                  0.5 * ((target_y - y_pred_mean) ** 2) / y_pred_var
            nll = nll.mean()
            loss = nll
        else:
            nll = None
            loss = None
        return y_pred_mean, y_pred_var, loss, None, nll  # kl always None
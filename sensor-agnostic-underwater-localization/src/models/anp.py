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


# LatentModel:

class LatentModel(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_latent=num_hidden,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout)
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=input_dim,
                                                          output_dim=output_dim,
                                                          dropout=dropout)
        self.decoder = Decoder(num_hidden,
                               input_dim=input_dim,
                               output_dim=output_dim,
                               dropout=dropout)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0,
                predict_with_prior: bool = False):
        # predict_with_prior: if True the DECODER is driven by the PRIOR latent
        # (context only) even when target_y is supplied -- the deployment-faithful
        # path (at inference the latent cannot peek at target labels). The
        # posterior + KL/NLL are still computed when target_y is given, so the
        # validation loss stays comparable to training; only the z that drives the
        # prediction changes. Use False for training (posterior teacher forcing).
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
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1):
        super(DeterministicModel, self).__init__()
        self.deterministic_encoder = DeterministicEncoder(num_hidden,
                                                          num_latent=num_hidden,
                                                          input_dim=input_dim,
                                                          output_dim=output_dim,
                                                          dropout=dropout)
        self.decoder = DeterministicDecoder(num_hidden,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            dropout=dropout)

    def forward(self, context_x, context_y, target_x, target_y=None, beta: float = 1.0,
                predict_with_prior: bool = False):
        # predict_with_prior is accepted for a uniform interface with the latent
        # models but has no effect: a deterministic CNP never uses target labels
        # to predict, so there is no posterior to peek at.
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
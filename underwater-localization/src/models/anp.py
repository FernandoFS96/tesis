import math
import torch as t
import torch.nn as nn

# src/models/anp.py: Define ANP components

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
        #self.input_projection = Linear(input_dim + 3, num_hidden)
        self.input_projection = Linear(input_dim + output_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        encoder_input = t.cat([x, y], dim=-1)
        encoder_input = self.input_projection(encoder_input)
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        log_sigma = 1 * t.tanh(log_sigma)

        std = t.exp(0.5 * log_sigma)
        std = t.clamp(std, min=1e-6, max=1e6)
        eps = t.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, log_sigma, z

class DeterministicEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim, output_dim):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        #self.input_projection = Linear(input_dim + 3, num_hidden)
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
        #self.mean_projection = Linear(num_hidden * 3, 3)
        #self.log_var_projection = Linear(num_hidden * 3, 3)
        self.mean_projection    = Linear(num_hidden*3, output_dim)
        self.log_var_projection = Linear(num_hidden*3, output_dim)


    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        target_x = self.target_projection(target_x)
        hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)
        for linear in self.linears:
            hidden = t.relu(linear(hidden))
        mean = self.mean_projection(hidden)
        var = 10000 * t.sigmoid(self.log_var_projection(hidden))
        return mean, var

class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_k):
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        attn = t.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        result = t.bmm(attn, value)
        return result, attn

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

# LatentModel: Define in src/models/anp.py

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

    def forward(self, context_x, context_y, target_x, target_y=None):
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
            loss = nll + kl
        else:
            kl = None
            loss = None
            nll = None

        return y_pred_mean, y_pred_var, loss, kl, nll

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

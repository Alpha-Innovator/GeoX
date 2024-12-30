import torch
import torch.nn as nn
import re
import os
import torch.nn.functional as F

class VoteGenerator(nn.Module):

    def __init__(self, dim, nhead, nlayers, n_query):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.attention = nn.Linear(dim, 1)
        self.fc = nn.Linear(dim, dim)
        self.n_query = n_query

    def forward(self, image_embeds):
        image_embeds = image_embeds.permute(1, 0, 2)
        transformed = self.transformer_encoder(image_embeds)
        transformed = transformed.permute(1, 0, 2)
        attention_scores = self.attention(transformed).softmax(dim=1)
        pooled = (transformed * attention_scores).sum(dim=1, keepdim=True)
        pooled = pooled.expand(-1, self.n_query, -1)
        query = self.fc(pooled)
        return query


def build_vote_generator(dim=768, nhead=8, nlayers=2, n_query=8):
    return VoteGenerator(dim=dim, nhead=nhead, nlayers=nlayers, n_query=n_query)


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    raise ValueError(f'Unknown projector type: {projector_type}')


class Sampler(nn.Module):
    def __init__(self, embed_dim, tau):
        super(Sampler, self).__init__()
        self.linear_layer = nn.Linear(embed_dim, 2)
        self.tau = tau
        nn.init.kaiming_normal_(self.linear_layer.weight)
        self.linear_layer.bias.data.fill_(0)

    def forward(self, x):
        logits = self.linear_layer(x)
        gumbel_logits = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        selected_probs = gumbel_logits[..., 1]
        return selected_probs


def build_sampler(embed_dim=768, tau=0.1, sampler_layers=[3, 6, 9]):

    samplers = []
    for i in range(12):  # Ensure enough layers are created
        if i in sampler_layers:
            samplers.append(Sampler(embed_dim, tau))
        else:
            samplers.append(None)  # Use None for non-sampler layers
    return nn.ModuleList(samplers)

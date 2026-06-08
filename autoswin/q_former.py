import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    def __init__(self, query_dim, encoder_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim, kdim=encoder_dim, vdim=encoder_dim, num_heads=num_heads, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim)
        )
        self.norm3 = nn.LayerNorm(query_dim)

    def forward(self, queries, encoder_outputs):
        # Cross-attention: queries attend to encoder outputs
        q, _ = self.cross_attn(queries, encoder_outputs, encoder_outputs)
        queries = queries + q
        queries = self.norm1(queries)
        # Self-attention: queries attend to themselves
        q, _ = self.self_attn(queries, queries, queries)
        queries = queries + q
        queries = self.norm2(queries)
        # MLP
        q = self.mlp(queries)
        queries = queries + q
        queries = self.norm3(queries)
        return queries


class QFormer(nn.Module):
    def __init__(self, num_query_tokens=32, query_dim=256, encoder_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, query_dim))
        self.input_proj = nn.Linear(encoder_dim, query_dim)
        self.layers = nn.ModuleList([
            QFormerLayer(query_dim, query_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq, encoder_dim)
        batch_size = encoder_outputs.size(0)
        encoder_outputs = self.input_proj(encoder_outputs)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_outputs)
        return queries  # (batch, num_query_tokens, query_dim)


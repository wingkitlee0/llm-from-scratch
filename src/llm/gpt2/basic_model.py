import torch
from torch import Tensor, nn


class CasualAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout: float, qkv_bias: bool = False
    ):
        super().__init__()
        self.d_out = d_out
        self.W_qry = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_val = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_qry(x)
        values = self.W_val(x)

        attn_scores = queries @ keys.transpose(-2, -1)

        # underscore ops indicate in-place operations
        attn_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens],
            -torch.inf,
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )
        attn_weights = self.dropout(attn_weights)

        context_vectors = attn_weights @ values
        return context_vectors


class MultiHeaddAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)

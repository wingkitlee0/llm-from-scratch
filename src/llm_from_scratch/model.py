from torch import dtype, nn, Tensor
import torch


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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_qry = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_val = nn.Linear(d_in, d_out, bias=qkv_bias)
        # A linear layer to combine head outputs (optional)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape
        # shape = (b, num_tokens, d_out)
        queries = self.W_qry(x)
        keys = self.W_key(x)
        values = self.W_val(x)

        # Recall d_out = num_heads * head_dim
        # So adding a num_heads dimension such that
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose(-2, -1) -> (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention score is about pairwise token similarity
        # Matmul -> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Apply mask in-place. mask=1 -> -torch.inf
        # -torch.inf is used because softmax(x) -> 0 when x -> -inf
        attn_scores.masked_fill_(~mask_bool, -torch.inf)

        # Apply softmax
        # dim=-1 -> normalize over the last dimension
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )
        # Apply dropout mask
        attn_weights = self.dropout(attn_weights)

        # Matmul -> (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vectors = (attn_weights @ values).transpose(1, 2)
        # Reshape -> (b, num_tokens, d_out), since d_out = num_heads * head_dim
        context_vectors = context_vectors.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vectors)

        attn_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens],
            -torch.inf,
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )

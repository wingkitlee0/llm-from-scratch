import torch
import torch.nn as nn
from torch import Tensor


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


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Use GPT-2 original variance calculation
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return norm_x * self.scale + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int], use_shortcut: bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)  # Pre-layer Norm
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut
        return x

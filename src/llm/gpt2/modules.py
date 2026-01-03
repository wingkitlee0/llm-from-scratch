import torch
import torch.nn as nn
import torch.nn.functional as F
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
        num_kv_heads: int | None = None,
        enable_flash_att: bool = True,
        enable_gqa: bool = False,
    ):
        """Multi-Head Attention with optional Flash Attention and Grouped Query Attention.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            context_length: Maximum sequence length
            num_heads: Number of query heads
            dropout: Dropout rate
            qkv_bias: Whether to use bias in Q/K/V projections
            num_kv_heads: Number of key/value heads for GQA (only used if enable_gqa=True)
            enable_flash_att: Whether to use Flash Attention (via scaled_dot_product_attention)
            enable_gqa: Whether to enable Grouped Query Attention
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.enable_flash_att = enable_flash_att
        self.enable_gqa = enable_gqa

        # Determine number of KV heads
        if enable_gqa:
            if num_kv_heads is None:
                raise ValueError("num_kv_heads must be specified when enable_gqa=True")
            assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
            self.num_kv_heads = num_kv_heads
        else:
            # Standard MHA: same number of KV heads as query heads
            self.num_kv_heads = num_heads

        # Query projection (always full size)
        self.W_qry = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Key/Value projections
        if enable_gqa:
            # GQA: smaller KV projections
            kv_dim = self.num_kv_heads * self.head_dim
            self.W_key = nn.Linear(d_in, kv_dim, bias=qkv_bias)
            self.W_val = nn.Linear(d_in, kv_dim, bias=qkv_bias)
        else:
            # Standard MHA: same size as queries
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_val = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection
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

        # Project to Q, K, V
        queries = self.W_qry(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_val(x)

        # Reshape queries: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Reshape keys/values based on GQA or MHA
        if self.enable_gqa:
            # GQA: (b, num_tokens, kv_dim) -> (b, num_tokens, num_kv_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Standard MHA: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.enable_flash_att:
            # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
            context_vectors = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
                scale=self.head_dim ** -0.5,
                enable_gqa=self.enable_gqa,
            )
        else:
            # Standard attention implementation
            # Expand KV heads if using GQA
            if self.enable_gqa:
                num_queries_per_kv = self.num_heads // self.num_kv_heads
                keys = keys.repeat_interleave(num_queries_per_kv, dim=1)
                values = values.repeat_interleave(num_queries_per_kv, dim=1)

            # Attention score is about pairwise token similarity
            # Matmul -> (b, num_heads, num_tokens, num_tokens)
            attn_scores = queries @ keys.transpose(-2, -1)
            mask_bool = self.mask[:num_tokens, :num_tokens].bool()

            # Apply mask in-place. mask=1 -> -torch.inf
            # -torch.inf is used because softmax(x) -> 0 when x -> -inf
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            # Apply softmax
            # dim=-1 -> normalize over the last dimension
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5,
                dim=-1,
            )
            # Apply dropout mask
            attn_weights = self.dropout(attn_weights)

            # Matmul -> (b, num_heads, num_tokens, head_dim)
            context_vectors = attn_weights @ values

        # Transpose back: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vectors = context_vectors.transpose(1, 2)
        # Reshape -> (b, num_tokens, d_out), since d_out = num_heads * head_dim
        context_vectors = context_vectors.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vectors)


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
            num_kv_heads=cfg.get("num_kv_heads"),
            enable_flash_att=cfg.get("enable_flash_att", True),
            enable_gqa=cfg.get("enable_gqa", False),
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

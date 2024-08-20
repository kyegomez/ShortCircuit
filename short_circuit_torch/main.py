from zeta.nn import MultiQueryAttention, FeedForward
from torch import nn, Tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.heads = heads
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )

        # FFN
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swish=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Apply normalization to the input tensor
        x = self.norm(x)

        # Apply attention mechanism to the tensor
        # We only need the first output of self.attn, so we ignore the rest with '_'
        x, _, _ = self.attn(x)

        # Add the original input (residual connection) to the output of the attention mechanism
        x += self.norm(x)

        # Apply feed-forward network to the tensor and normalize it
        x = self.norm(self.ffn(x))

        # Add the output of the attention mechanism (residual connection) to the output of the feed-forward network
        x += self.norm(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dropout,
                    heads,
                    dim_head,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PolicyModule(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head=dim // heads,
            mlp_dim=dim * 4,
            dropout=0.1,
        )

        # Attn
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _, _ = self.attn(x)
        x = self.transformer(x)
        return x


class ValueModule(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
    ):
        super().__init__()
        # Transformer layers for the value module -> cross attention -> linear
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head=dim // heads,
            mlp_dim=dim * 4,
            dropout=0.1,
        )

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )

        # Linear Layer
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x, _, _ = self.attn(x)
        x = self.transformer(x)
        x = self.linear(x)
        return x


class ShortCircuitNet(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Transformer
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head=dim // heads,
            mlp_dim=dim * 4,
            dropout=0.1,
        )

        # Policy Module
        self.policy_module = PolicyModule(
            dim,
            heads,
            depth,
        )

        # Value Module
        self.value_module = ValueModule(
            dim,
            heads,
            depth,
        )

        # Softmax
        self.softmx_act = nn.Softmax(dim=-1)

        # Tanh Layer
        self.tanh_act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # Transformer  -> split path way -> policy module -> value module
        x = self.transformer(x)

        # Policy Module
        policy = self.policy_module(x)

        # Value Module
        value = self.value_module(x)

        # Softmax
        softed = self.softmx_act(policy)

        # Tanh
        self.tanded = self.tanh_act(value)

        return softed, self.tanded

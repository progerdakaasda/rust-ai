from dataclasses import dataclass


@dataclass
class GPTConfig:

    vocab_size: int = 32000

    context_length: int = 1024

    embedding_dim: int = 640

    num_layers: int = 10

    num_heads: int = 10

    mlp_hidden_dim: int = 2560

    dropout: float = 0.0

    bias: bool = False

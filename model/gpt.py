import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig



class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        assert config.embedding_dim % config.num_heads == 0


        self.num_heads = config.num_heads

        self.head_dim = (
            config.embedding_dim //
            config.num_heads
        )


        self.qkv = nn.Linear(
            config.embedding_dim,
            config.embedding_dim * 3,
            bias=config.bias
        )


        self.out = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias
        )


        self.dropout = nn.Dropout(
            config.dropout
        )


        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length
                )
            ).view(
                1,
                1,
                config.context_length,
                config.context_length
            )
        )



    def forward(self, x):

        B, T, C = x.shape


        qkv = self.qkv(x)


        q, k, v = qkv.chunk(
            3,
            dim=-1
        )


        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        attention = (
            q @ k.transpose(-2, -1)
            /
            (self.head_dim ** 0.5)
        )


        attention = attention.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )


        attention = F.softmax(
            attention,
            dim=-1
        )


        attention = self.dropout(
            attention
        )


        y = attention @ v


        y = y.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )


        return self.out(y)




class MLP(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                config.embedding_dim,
                config.mlp_hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                config.mlp_hidden_dim,
                config.embedding_dim
            ),

            nn.Dropout(
                config.dropout
            )
        )



    def forward(self,x):

        return self.net(x)





class Block(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        self.norm1 = nn.LayerNorm(
            config.embedding_dim
        )


        self.attn = CausalSelfAttention(
            config
        )


        self.norm2 = nn.LayerNorm(
            config.embedding_dim
        )


        self.mlp = MLP(
            config
        )



    def forward(self,x):

        x = x + self.attn(
            self.norm1(x)
        )


        x = x + self.mlp(
            self.norm2(x)
        )


        return x





class GPT(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        self.config = config


        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )


        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim
        )


        self.blocks = nn.ModuleList(
            [
                Block(config)
                for _ in range(config.num_layers)
            ]
        )


        self.norm = nn.LayerNorm(
            config.embedding_dim
        )


        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )


        # tie weights
        self.lm_head.weight = (
            self.token_embedding.weight
        )



    def forward(
        self,
        idx
    ):

        B, T = idx.shape


        positions = torch.arange(
            T,
            device=idx.device
        )


        x = (
            self.token_embedding(idx)
            +
            self.position_embedding(positions)
        )


        for block in self.blocks:

            x = block(x)


        x = self.norm(x)


        logits = self.lm_head(x)


        return logits

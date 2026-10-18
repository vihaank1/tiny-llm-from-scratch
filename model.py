import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()

        self.head_size = head_size
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)

        length = x.shape[1]
        mask = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=x.device)
        )

        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        return weights @ v
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")
        
        head_size=embedding_size // num_heads

        self.heads=nn.ModuleList([
            AttentionHead(embedding_size, head_size)
            for _ in range(num_heads)
        ])

        self.projection=nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        head_outputs=[head(x) for head in self.heads]
        combined=torch.cat(head_outputs, dim=-1)
        return self.projection(combined)
    

class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.expand=nn.Linear(embedding_size, 4 * embedding_size)
        self.activation=nn.GELU()
        self.project=nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x):
        expanded=self.expand(x)
        activated=self.activation(expanded)
        return self.project(activated)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        self.norm1=nn.LayerNorm(embedding_size)
        self.attention=MultiHeadAttention(embedding_size, num_heads)

        self.norm2=nn.LayerNorm(embedding_size)
        self.feed_forward=FeedForward(embedding_size)
        self.dropout=nn.Dropout(0.2)

    def forward(self,x):
        attention_output=self.attention(self.norm1(x))
        x=x+self.attention(self.norm1(x))

        feed_forward_output=self.feed_forward(self.norm2(x))
        x=x+self.feed_forward(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
            self, vocab_size, block_size,
            embedding_size=16, num_heads=2, num_layers=2
    ):
        super().__init__()

        self.block_size=block_size
        self.token_embedding=nn.Embedding(vocab_size, embedding_size)
        self.position_embedding=nn.Embedding(block_size, embedding_size)

        self.blocks=nn.ModuleList([
            TransformerBlock(embedding_size, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm=nn.LayerNorm(embedding_size)
        self.output_layer=nn.Linear(embedding_size, vocab_size)

    def forward(self, token_ids):
        sequence_length=token_ids.shape[1]

        if sequence_length > self.block_size:
            raise ValueError("Sequence exceeds block_size")
        
        positions=torch.arange(sequence_length, device=token_ids.device)

        x=(
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x=block(x)
            
        x=self.final_norm(x)
        logits=self.output_layer(x)
        return logits
    
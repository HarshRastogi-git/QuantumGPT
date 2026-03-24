import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)[0]
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x

class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout):
        super(QuantumGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = self.fc_out(x)
        return x

    def generate(self, x, gen_length):
        for _ in range(gen_length):
            x = torch.cat((x, self.forward(x).argmax(dim=-1)[:, -1].unsqueeze(1)), dim=1)
        return x

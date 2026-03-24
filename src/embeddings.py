import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.encoding = self.create_positional_encoding(embed_size, max_len)

    def create_positional_encoding(self, embed_size, max_len):
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(embed_size, dtype=torch.float) / embed_size * 2 * torch.pi

        pos_enc = pos / (10000 ** (i / embed_size))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=1)
        return pos_enc.unsqueeze(0)  # (1, max_len, embed_size)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]

# Example of usage
# token_embedding_layer = TokenEmbeddings(vocab_size=10000, embed_size=512)
# positional_embedding_layer = PositionalEmbedding(embed_size=512)
# token_ids = torch.randint(0, 10000, (32, 100))  # Example token IDs
# token_embeddings = token_embedding_layer(token_ids)
# positional_embeddings = positional_embedding_layer(token_embeddings)
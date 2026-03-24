class TransformerBlock:
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        self.attention = MultiHeadAttention(heads, embed_size)
        self.norm1 = LayerNormalization(embed_size)
        self.norm2 = LayerNormalization(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)
        x = self.dropout1(self.norm1(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout2(self.norm2(forward + x))
        return x
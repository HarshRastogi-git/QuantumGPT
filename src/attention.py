import torch
import torch.nn.functional as F

class ScaledDotProductAttention:
    def __init__(self, dropout=0.1):
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        d_k = query.size(-1)
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, value)
        return output, attn

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout=0.1):
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(dropout)

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        output, attn = self.attention.forward(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_head)
        output = self.fc(output)
        output = self.dropout(output)
        return output, attn
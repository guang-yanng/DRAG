import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        # self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class Attention(nn.Module):
    def __init__(self, emb_size):
        super(Attention, self).__init__()
        self.weight_W = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
    def forward(self, inputs, masks=None):
        batch_size = inputs.size(0)
        scores = torch.m
import torch
import torch.nn as nn
import math

def generate_padding_mask(input_ids, pad_idx):
    # (B, L)
    mask = (input_ids != pad_idx)
    # (B, 1, 1, L)
    return mask.unsqueeze(1).unsqueeze(2)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.d_k = d_model //head
        self.d_q  = d_model //head
        self.d_v  = d_model //head
        self.fc_q = nn.Linear(d_model, head * self.d_q)
        self.fc_k = nn.Linear(d_model, head * self.d_k)
        self.fc_v = nn.Linear(d_model, head * self.d_v)
        self.fc_o = nn.Linear(head * self.d_v, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, attention_mask: torch.Tensor = None):
        b_s, len_q = queries.shape[:2]
        len_k = keys.shape[1]
        
        q = self.fc_q(queries).view(b_s, len_q, self.head, self.d_q).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, len_k, self.head, self.d_k).permute(0, 2, 1, 3)
        v = self.fc_v(values).view(b_s, len_k, self.head, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, -1e4)
        
        att = torch.softmax(att, dim=-1)
        output = torch.matmul(att, v)                       # (B, head, L, d_k)
        output = output.permute(0, 2, 1, 3).contiguous()     # (B, L, head, d_k)
        output = output.view(b_s, len_q, self.d_model)       # concat heads

        output = self.fc_o(output)        # output projection
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(- math.log(10000.0)/d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe[:, :x.size(1)]
        pe = pe.expand(x.size(0), -1, -1)
        x = x + pe
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model:int, head: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = ScaledDotProductAttention(head, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, attention_mask):
        # Self-attention
        attn_out = self.self_attention(src, src, src, attention_mask)
        src = src + self.dropout1(attn_out)
        src = self.layer_norm1(src)

        # Feed-forward
        ff_out = self.feed_forward(src)
        src = src + self.dropout2(ff_out)
        src = self.layer_norm2(src)

        return src


        
class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, head, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        features = inputs
        for layer in self.layers:
            features = layer(features, attention_mask)
        
        return features

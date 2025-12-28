import torch
import torch.nn as nn
from transformer import PositionalEncoding, TransformerEncoder, generate_padding_mask

class TransformerModel(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout, vocab_size, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln_head = nn.Linear(d_model, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.pad_idx = pad_idx
        
    def forward(self, input_ids, labels=None):
        attention_mask = generate_padding_mask(input_ids, self.pad_idx)
        input_embeddings = self.embedding(input_ids)
        features = self.positional_encoding(input_embeddings)
        features = self.encoder(features, attention_mask)
        
        # Pooling strategy
        features = features[:, 0]  # hoặc features.mean(dim=1)
        features = self.dropout(features)
        logits = self.ln_head(features)
        
        loss = self.loss(logits, labels) if labels is not None else None
        return logits, loss

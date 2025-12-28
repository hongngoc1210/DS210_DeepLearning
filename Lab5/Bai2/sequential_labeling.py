import torch
import torch.nn as nn
from transformer import PositionalEncoding, TransformerEncoder, generate_padding_mask

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, d_model: int, 
             n_heads: int, n_layers: int, d_ff: int, dropout: float, pad_idx: int):
        super().__init__()
        self.num_labels = num_tags
        
        #token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        
        self.tkn_classifier = nn.Linear(d_model, num_tags)
        self.dropout = nn.Dropout(dropout)
        #token level loss
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.pad_idx = pad_idx
        
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(input_ids, self.pad_idx)
        
        input_embeddings = self.embedding(input_ids)
        features = self.positional_encoding(input_embeddings)
        features = self.encoder(features, attention_mask) # B, L, d_model
        
        logits =self.tkn_classifier(self.dropout(features))  # B, L, num_labels
        
        if labels is not None:
            loss = self.loss(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            return logits, loss
        else:
            return logits, None
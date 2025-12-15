# biLSTM.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    """
    BiLSTM encoder that outputs per-token logits for NER.
    Returns logits of shape (batch_size, seq_len, num_tags).
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 num_tags, dropout=0.3, padding_idx=0):
        super(BiLSTMEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x, lengths):
        """
        x: LongTensor [B, L]
        lengths: LongTensor [B] (actual lengths before padding)
        returns logits: FloatTensor [B, L, num_tags]
        """
        embedded = self.embedding(x)           # [B, L, E]
        embedded = self.dropout(embedded)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, L, H*2]

        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # [B, L, num_tags]
        return logits

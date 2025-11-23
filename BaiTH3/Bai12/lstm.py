import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    """
    5-layer LSTM classifier (configurable).
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256,
                 num_layers=5, num_classes=3, dropout=0.5, bidirectional=True, padding_idx=0):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        fc_input_size = hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(fc_input_size)
        self.fc1 = nn.Linear(fc_input_size, fc_input_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_input_size // 2, num_classes)

    def forward(self, x, lengths):
        """
        x: [B, L]
        lengths: [B]
        """
        embedded = self.embedding(x)  # [B, L, E]
        embedded = self.dropout(embedded)

        # pack
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H*num_dir]

        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]  # last layer forward
            hidden_backward = hidden[-1, :, :]  # last layer backward
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)  # [B, H*2]
        else:
            final_hidden = hidden[-1, :, :]  # [B, H]

        final_hidden = self.layer_norm(final_hidden)
        out = self.dropout(final_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

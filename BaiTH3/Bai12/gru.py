import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256,
                 num_layers=5, num_classes=3, dropout=0.3, bidirectional=True, padding_idx=0):
        super(GRUClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        fc_input_size = hidden_size * self.num_directions

        # Attention layer
        self.attention = nn.Linear(fc_input_size, 1)

        self.layer_norm = nn.LayerNorm(fc_input_size)
        self.fc1 = nn.Linear(fc_input_size, fc_input_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_input_size // 2, num_classes)

    def forward(self, x, lengths):
        """
        x: [B, L]
        lengths: [B]
        """
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H*D]

        # attention scores
        attn_scores = self.attention(output).squeeze(-1)  # [B, L]
        # mask padding
        max_len = output.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, L]
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)  # [B, H*D]

        # use hidden states: combine both
        final_hidden = context  # use attention context as representation

        final_hidden = self.layer_norm(final_hidden)
        out = self.dropout(final_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits
        
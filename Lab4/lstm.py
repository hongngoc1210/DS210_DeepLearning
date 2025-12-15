import torch
import torch.nn as nn
from typing import Tuple, List
from vocab import Vocab

class Seq2SeqLSTM(nn.Module):
    def __init__(self, d_model: int, n_encoder: int, 
                 n_decoder: int, dropout: float, vocab: Vocab):
        super().__init__()
        
        self.vocab = vocab
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens,
            embedding_dim=d_model,
            padding_idx=vocab.pad_idx
        )
        
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder,
            batch_first=True,
            dropout=dropout if n_encoder > 1 else 0,
            bidirectional=True
        )
        
        self.tgt_embedding = nn.Embedding(
            vocab.total_tgt_tokens,
            d_model,
            padding_idx=vocab.pad_idx
        )
        
        # Project bidirectional hidden states to decoder hidden size
        self.bridge_h = nn.Linear(d_model * 2, d_model)
        self.bridge_c = nn.Linear(d_model * 2, d_model)
        
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_decoder,
            batch_first=True,
            dropout=dropout if n_decoder > 1 else 0,
            bidirectional=False
        )

        self.output = nn.Linear(d_model, vocab.total_tgt_tokens)
        
        # Loss will be computed outside
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: source sequences [batch_size, src_len]
            y: target sequences [batch_size, tgt_len] (including <bos>)
        Returns:
            outputs: logits [batch_size, tgt_len-1, vocab_size]
            loss: scalar loss value
        """
        bs = x.size(0)
        vocab_size = self.output.out_features
        
        # Encode
        embedded_x = self.src_embedding(x)
        _, (hidden_states, cell_states) = self.encoder(embedded_x)
        
        # hidden_states: [num_layers*2, bs, d_model]
        # Reshape to separate directions
        hidden_states = hidden_states.view(
            self.encoder.num_layers, 2, bs, self.d_model
        )  # [num_layers, 2, bs, d_model]
        
        cell_states = cell_states.view(
            self.encoder.num_layers, 2, bs, self.d_model
        )
        
        # Concatenate forward and backward for each layer
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()  # [num_layers, bs, 2, d_model]
        hidden_states = hidden_states.view(self.encoder.num_layers, bs, self.d_model * 2)
        
        cell_states = cell_states.permute(0, 2, 1, 3).contiguous()
        cell_states = cell_states.view(self.encoder.num_layers, bs, self.d_model * 2)
        
        # Project to decoder dimensions
        hidden_states = self.bridge_h(hidden_states)  # [num_layers, bs, d_model]
        cell_states = self.bridge_c(cell_states)  # [num_layers, bs, d_model]
        
        # Teacher forcing: decode all positions at once
        tgt_len = y.size(1) - 1  # Exclude last token for input
        outputs = []
        
        for t in range(tgt_len):
            y_t = y[:, t].unsqueeze(1)  # [bs, 1]
            embedded_y = self.tgt_embedding(y_t)  # [bs, 1, d_model]
            
            # Decode one step
            _, (hidden_states, cell_states) = self.decoder(
                embedded_y, (hidden_states, cell_states)
            )
            
            # Get last layer hidden state
            last_hidden = hidden_states[-1]  # [bs, d_model]
            logits = self.output(last_hidden)  # [bs, vocab_size]
            outputs.append(logits)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [bs, tgt_len, vocab_size]
        
        # Compute loss
        loss = self.loss_fn(
            outputs.reshape(-1, vocab_size),
            y[:, 1:].reshape(-1)
        )
        
        return outputs, loss
    
    def predict(self, x: torch.Tensor, max_len: int = 100) -> List[torch.Tensor]:
        """
        Args:
            x: source sequences [batch_size, src_len]
            max_len: maximum generation length
        Returns:
            outputs: list of predicted token ids, each [batch_size, 1]
        """
        bs = x.size(0)
        
        # Encode
        embedded_x = self.src_embedding(x)
        _, (hidden_states, cell_states) = self.encoder(embedded_x)
        
        # Process encoder states
        hidden_states = hidden_states.view(
            self.encoder.num_layers, 2, bs, self.d_model
        )
        cell_states = cell_states.view(
            self.encoder.num_layers, 2, bs, self.d_model
        )
        
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(self.encoder.num_layers, bs, self.d_model * 2)
        
        cell_states = cell_states.permute(0, 2, 1, 3).contiguous()
        cell_states = cell_states.view(self.encoder.num_layers, bs, self.d_model * 2)
        
        # Project to decoder dimensions
        hidden_states = self.bridge_h(hidden_states)
        cell_states = self.bridge_c(cell_states)
        
        # Start with <bos> token
        y_t = torch.full(
            (bs, 1),
            self.vocab.bos_idx,
            dtype=torch.long,
            device=x.device
        )
        
        outputs = []
        for _ in range(max_len):
            embedded_y = self.tgt_embedding(y_t)
            
            # Decode one step
            _, (hidden_states, cell_states) = self.decoder(
                embedded_y, (hidden_states, cell_states)
            )
            
            # Get predictions
            last_hidden = hidden_states[-1]
            logits = self.output(last_hidden)
            y_t = logits.argmax(dim=-1, keepdim=True)  # [bs, 1]
            
            outputs.append(y_t)
            
            # Check if all sequences have generated <eos>
            if (y_t == self.vocab.eos_idx).all():
                break
        
        return outputs
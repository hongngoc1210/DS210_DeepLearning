import torch
import torch.nn as nn
from typing import Tuple
from vocab import Vocab

class Seq2SeqLSTM(nn.Module):
    def __init__(self, d_model: int, n_encoder: int, 
                 n_decoder: int, dropout: float, vocab: Vocab):
        super().__init__()  
        self.vocab = vocab
        self.d_model = d_model
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        
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
        self.src_dropout = nn.Dropout(dropout)
        
        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens,
            embedding_dim=2*d_model,
            padding_idx=vocab.pad_idx
        )
        
        self.decoder = nn.LSTM(
            input_size=2*d_model,
            hidden_size=2*d_model,
            num_layers=n_decoder,
            batch_first=True,  
            dropout=dropout if n_decoder > 1 else 0,
            bidirectional=False 
        )
        self.tgt_dropout = nn.Dropout(dropout)
        
        # Project bidirectional encoder to decoder
        self.bridge_h = nn.Linear(2*d_model, 2*d_model)
        self.bridge_c = nn.Linear(2*d_model, 2*d_model)
        
        self.attention_concat = nn.Linear(4*d_model, 2*d_model)  
        self.output_head = nn.Linear(
            in_features=2*d_model,
            out_features=vocab.total_tgt_tokens
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    def encode(self, x: torch.Tensor):
        """Encoder: 3-layer bidirectional LSTM

        Args:
            x: (bs, src_len)
        Returns:
            encoder_outputs: (bs, src_len, 2*d_model)
            hidden: (n_decoder, bs, 2*d_model)
            cell: (n_decoder, bs, 2*d_model)
        """
        embedding_x = self.src_dropout(self.src_embedding(x))
        encoder_outputs, (hidden, cell) = self.encoder(embedding_x)
        
        # hidden/cell: [n_encoder*2, bs, d_model]
        # Reshape and project to decoder dimensions
        bs = x.size(0)
        hidden = hidden.view(self.n_encoder, 2, bs, self.d_model)
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        hidden = hidden.view(self.n_encoder, bs, 2*self.d_model)
        hidden = self.bridge_h(hidden)  # [n_encoder, bs, 2*d_model]
        
        cell = cell.view(self.n_encoder, 2, bs, self.d_model)
        cell = cell.permute(0, 2, 1, 3).contiguous()
        cell = cell.view(self.n_encoder, bs, 2*self.d_model)
        cell = self.bridge_c(cell)  # [n_encoder, bs, 2*d_model]
        
        # Use only first n_decoder layers if encoder has more layers
        if self.n_encoder > self.n_decoder:
            hidden = hidden[:self.n_decoder]
            cell = cell[:self.n_decoder]
        # Repeat if decoder has more layers
        elif self.n_decoder > self.n_encoder:
            repeat_factor = self.n_decoder // self.n_encoder
            hidden = hidden.repeat(repeat_factor, 1, 1)[:self.n_decoder]
            cell = cell.repeat(repeat_factor, 1, 1)[:self.n_decoder]
        
        return encoder_outputs, hidden, cell
    
    def attention(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """Global Attention, Alignment Mechanism

        Args:
            decoder_hidden: (bs, 2*d_model)
            encoder_outputs: (bs, src_len, 2*d_model)
        Returns:
            context_vector: (bs, 2*d_model)
            attn_weights: (bs, src_len)
        """
        # Compute alignment scores
        decoder_hidden = decoder_hidden.unsqueeze(2)  # [bs, 2*d_model, 1]
        scores = torch.bmm(encoder_outputs, decoder_hidden)  # [bs, src_len, 1]
        scores = scores.squeeze(2)  # [bs, src_len]
        
        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=1)
        
        # Compute context vector as weighted sum
        attention_weights_expanded = attention_weights.unsqueeze(1)  # [bs, 1, src_len]
        context_vector = torch.bmm(attention_weights_expanded, encoder_outputs)  # [bs, 1, 2*d_model]
        context_vector = context_vector.squeeze(1)  # [bs, 2*d_model]
        
        return context_vector, attention_weights
    
    def decode_step(self, input_token: torch.Tensor, encoder_outputs: torch.Tensor,
                    hidden: torch.Tensor, cell: torch.Tensor):
        """Single decoder step with attention
        
        Args:
            input_token: (bs,) or (bs, 1)
            encoder_outputs: (bs, src_len, 2*d_model)
            hidden: (n_decoder, bs, 2*d_model)
            cell: (n_decoder, bs, 2*d_model)
        Returns:
            output: (bs, vocab_size)
            hidden: (n_decoder, bs, 2*d_model)
            cell: (n_decoder, bs, 2*d_model)
            attention_weights: (bs, src_len)
        """
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)  # [bs, 1]
            
        # Embedding
        embedding = self.tgt_dropout(self.tgt_embedding(input_token))  # [bs, 1, 2*d_model]
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.decoder(embedding, (hidden, cell))
        
        decoder_hidden = hidden[-1]  # [bs, 2*d_model]
        
        # Compute attention
        context_vector, attention_weights = self.attention(decoder_hidden, encoder_outputs)
        
        # Concat context and hidden state
        lstm_out = lstm_out.squeeze(1)  # [bs, 2*d_model]
        concat_input = torch.cat([context_vector, lstm_out], dim=1)  # [bs, 4*d_model]
        
        # Compute attention hidden state
        attention_hidden = torch.tanh(self.attention_concat(concat_input))  # [bs, 2*d_model]
        
        # Output
        output = self.output_head(attention_hidden)  # [bs, vocab_size]
        
        return output, hidden, cell, attention_weights
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: source sequences (bs, src_len)
            y: target sequences (bs, tgt_len) including <bos>, excluding <eos>
        Returns:
            outputs: (bs, tgt_len, vocab_size)
            loss: scalar
        """
        batch_size = x.shape[0]
        tgt_len = y.shape[1]
        vocab_size = self.output_head.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=x.device)
        
        encoder_outputs, hidden, cell = self.encode(x)
        
        # Teacher forcing: use ground truth tokens
        for t in range(tgt_len):
            decoder_input = y[:, t]
            
            output, hidden, cell, _ = self.decode_step(
                decoder_input, encoder_outputs, hidden, cell)
            
            outputs[:, t, :] = output
        
        # Compute loss
        loss = self.loss(
            outputs[:, :-1].reshape(-1, vocab_size),
            y[:, 1:].reshape(-1)
        )
        
        return outputs, loss
    
    def predict(self, x: torch.Tensor, max_len: int = 100):
        """
        Args:
            x: source sequences (bs, src_len)
            max_len: maximum generation length
        Returns:
            preds: (bs, generated_len)
        """
        self.eval()
        bs = x.size(0)
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encode(x)
            
            # Start with <bos> token
            decoder_input = torch.full((bs,), self.vocab.bos_idx, dtype=torch.long, device=x.device)
            
            preds = []
            
            for _ in range(max_len):
                output, hidden, cell, _ = self.decode_step(
                    decoder_input, encoder_outputs, hidden, cell)
                
                pred_token = output.argmax(1)  # [bs]
                preds.append(pred_token.unsqueeze(1))  # [bs, 1]
                
                # Check if all sequences generated <eos>
                if (pred_token == self.vocab.eos_idx).all():
                    break
                    
                decoder_input = pred_token
            
            preds = torch.cat(preds, dim=1)  # [bs, generated_len]
                
        return preds
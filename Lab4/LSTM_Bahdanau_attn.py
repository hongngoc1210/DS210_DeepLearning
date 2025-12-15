import torch
import torch.nn as nn
from typing import Tuple, List
from vocab import Vocab

class Seq2seqLSTM(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_encoder: int, 
                 n_decoder: int,
                 dropout: float, 
                 vocab: Vocab
    ):
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

        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens, 
            embedding_dim=2*d_model, 
            padding_idx=vocab.pad_idx
        )

        self.attn_weights = nn.Linear(
            in_features=4*d_model,
            out_features=1
        )

        self.decoder = nn.LSTM(
            input_size=2*d_model,
            hidden_size=2*d_model,
            num_layers=n_decoder, 
            batch_first=True,
            dropout=dropout if n_decoder > 1 else 0,
            bidirectional=False
        )
        
        self.output_head = nn.Linear(
            in_features=2*d_model,
            out_features=vocab.total_tgt_tokens
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: source sequences [batch_size, src_len]
            y: target sequences [batch_size, tgt_len] (including <bos>, excluding <eos>)
        Returns:
            logits: [batch_size, tgt_len, vocab_size]
            loss: scalar loss value
        """
        embedded_x = self.src_embedding(x)
        bs, l, _ = embedded_x.shape
        
        # Encode all positions at once
        encoder_outputs, _ = self.encoder(embedded_x)  # [bs, l, 2*d_model]
        
        # Reshape for decoder: [n_decoder, bs, l, 2*d_model]
        encoder_outputs = encoder_outputs.unsqueeze(0).repeat(self.n_decoder, 1, 1, 1)
        
        # Initialize decoder hidden state
        dec_hidden_state = torch.zeros(
            self.n_decoder, bs, 2*self.d_model, device=x.device
        )
        
        # Teacher-forcing mechanism
        _, tgt_len = y.shape
        logits = []
        
        for ith in range(tgt_len):
            y_ith = y[:, ith].unsqueeze(-1)  # [bs, 1]
            dec_hidden_state = self.forward_step(
                y_ith, encoder_outputs, dec_hidden_state
            )
            
            # Get the last hidden states
            last_hidden_states = dec_hidden_state[-1]  # [bs, 2*d_model]
            logit = self.output_head(last_hidden_states)  # [bs, vocab_size]
            logits.append(logit.unsqueeze(1))
        
        logits = torch.cat(logits, dim=1)  # [bs, tgt_len, vocab_size]
        
        # Compute loss: compare logits with y shifted by 1
        loss = self.loss(
            logits[:, :-1].reshape(-1, self.vocab.total_tgt_tokens), 
            y[:, 1:].reshape(-1)
        )

        return logits, loss
    
    def aligning(self, query: torch.Tensor, k_v: torch.Tensor):
        """
        Bahdanau attention mechanism
        Args:
            query: [layers, bs, 2*d_model] - decoder hidden states
            k_v: [layers, bs, src_len, 2*d_model] - encoder outputs
        Returns:
            context: [layers, bs, 2*d_model] - attention-weighted context
        """
        _, _, l, _ = k_v.shape
        query = query.unsqueeze(2).repeat(1, 1, l, 1)  # [layers, bs, src_len, 2*d_model]

        # Concatenate and compute attention scores
        a = self.attn_weights(
            torch.cat([query, k_v], dim=-1)
        )  # [layers, bs, src_len, 1]
        
        a = torch.softmax(a, dim=2)  # [layers, bs, src_len, 1]
        
        # Weighted sum of encoder outputs
        context = (a * k_v).sum(dim=2)  # [layers, bs, 2*d_model]

        return context
    
    def forward_step(
            self, 
            input_ids: torch.Tensor, 
            enc_hidden_states: torch.Tensor,
            dec_hidden_state: torch.Tensor
        ):
        """
        Single decoding step
        Args:
            input_ids: [bs, 1] - current token
            enc_hidden_states: [n_decoder, bs, src_len, 2*d_model] - encoder outputs
            dec_hidden_state: [n_decoder, bs, 2*d_model] - previous decoder state
        Returns:
            dec_hidden_state: [n_decoder, bs, 2*d_model] - updated decoder state
        """
        embedded_input = self.tgt_embedding(input_ids)  # [bs, 1, 2*d_model]
        
        # Compute attention context
        cell_mem = self.aligning(dec_hidden_state, enc_hidden_states)
        
        # Decoder step
        _, (dec_hidden_state, cell_mem) = self.decoder(
            embedded_input, 
            (dec_hidden_state, cell_mem)
        )

        return dec_hidden_state.contiguous()

    def predict(self, x: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """
        Generate predictions
        Args:
            x: source sequences [batch_size, src_len]
            max_len: maximum generation length
        Returns:
            outputs: [batch_size, generated_len]
        """
        self.eval()
        embedded_x = self.src_embedding(x)
        bs, l, _ = embedded_x.shape
        
        # Encode
        encoder_outputs, _ = self.encoder(embedded_x)  # [bs, l, 2*d_model]
        
        # Reshape for decoder
        encoder_outputs = encoder_outputs.unsqueeze(0).repeat(
            self.n_decoder, 1, 1, 1
        )  # [n_decoder, bs, l, 2*d_model]
        
        # Initialize decoder hidden state
        dec_hidden_state = torch.zeros(
            self.n_decoder, bs, 2*self.d_model, device=x.device
        )
        
        # Start with <bos> token
        y_ith = torch.full((bs,), self.vocab.bos_idx, dtype=torch.long, device=x.device)
        
        outputs = []
        for _ in range(max_len):
            # Forward step
            dec_hidden_state = self.forward_step(
                y_ith.unsqueeze(-1), 
                encoder_outputs, 
                dec_hidden_state
            )
            
            # Get predictions
            last_hidden_states = dec_hidden_state[-1]  # [bs, 2*d_model]
            logit = self.output_head(last_hidden_states)  # [bs, vocab_size]
            
            y_ith = logit.argmax(dim=-1)  # [bs]
            outputs.append(y_ith.unsqueeze(-1))  # [bs, 1]
            
            # Check if all sequences have generated <eos>
            if (y_ith == self.vocab.eos_idx).all():
                break

        outputs = torch.cat(outputs, dim=-1)  # [bs, generated_len]
        
        return outputs
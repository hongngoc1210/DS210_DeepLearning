import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from rouge_score import rouge_scorer
import os

from vocab import Vocab
from dataset import PhoMTDataset, collate_fn
from lstm import Seq2SeqLSTM
# from LSTM_Bahdanau_attn import Seq2seqLSTM


# Load config
with open("lstm_config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Initialize vocab
vocab = Vocab(
    path=cfg["data_dir"],
    src_language="english",
    tgt_language="vietnamese"
)

# Dataset
train_set = PhoMTDataset(f"{cfg['data_dir']}/{cfg['train']}", vocab)
dev_set = PhoMTDataset(f"{cfg['data_dir']}/{cfg['dev']}", vocab)

train_loader = DataLoader(
    train_set,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_set,
    batch_size=cfg["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)

# Model
model = Seq2SeqLSTM(
    d_model=cfg["d_model"],
    n_encoder=cfg["n_encoder"],
    n_decoder=cfg["n_decoder"],
    dropout=cfg["dropout"],
    vocab=vocab
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer
optimizer = Adam(model.parameters(), lr=cfg["lr"])

# Training function
def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(train_loader)) as pbar:
        for it, batch in enumerate(train_loader):
            src = batch["english"].to(device)
            tgt = batch["vietnamese"].to(device)
            
            # Forward pass
            logits, loss = model(src, tgt[:, :-1])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Update running loss
            loss_val = loss.item()
            running_loss += loss_val
            
            # Update progress bar
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    
    mean_loss = running_loss / len(train_loader)
    return mean_loss

# Evaluation function
def eval_rouge(epoch):
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    
    with tqdm(desc='Epoch %d - Evaluating' % epoch, unit='it', total=len(dev_loader)) as pbar:
        for batch in dev_loader:
            src = batch["english"].to(device)
            tgt = batch["vietnamese"].to(device)
            
            with torch.no_grad():
                # Generate predictions
                preds = model.predict(src, max_len=cfg.get("MAX_LEN", 100))
                
                # Convert list of [bs, 1] tensors to [bs, seq_len]
                preds = torch.cat(preds, dim=1)
                
                # Decode to text
                pred_txt = vocab.decode_sentence(preds, "vietnamese")
                tgt_txt = vocab.decode_sentence(tgt, "vietnamese")
                
                # Calculate ROUGE-L for each pair
                for p, t in zip(pred_txt, tgt_txt):
                    if p.strip() and t.strip():
                        score = scorer.score(t, p)["rougeL"].fmeasure
                        scores.append(score)
            
            pbar.update()
    
    return sum(scores) / len(scores) if scores else 0.0

# Training loop
best_rouge = 0.0

for epoch in range(1, cfg["epochs"] + 1):
    # Train
    train_loss = train_epoch(epoch)
    
    # Evaluate
    rouge_l = eval_rouge(epoch)
    
    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, ROUGE-L: {rouge_l:.4f}")
    
    # Save best model only
    if rouge_l > best_rouge:
        best_rouge = rouge_l
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'rouge_l': rouge_l,
        }
        torch.save(checkpoint, "checkpoints/seq2seq_lstm_badahnau_best.pt")
        print(f"Saved best model with ROUGE-L: {rouge_l:.4f}")

print(f"\nTraining completed! Best ROUGE-L: {best_rouge:.4f}")
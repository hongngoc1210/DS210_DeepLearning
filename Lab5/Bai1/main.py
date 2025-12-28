import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os

from vocab import Vocab
from classification import TransformerModel
from viocd_dataset import ViOCDDataset, collate_fn

# Load config
with open("classification_config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Initialize vocab
vocab = Vocab(path=cfg["data_dir"])
print(f"Vocab size: {vocab.vocab_size}")

# Dataset
train_set = ViOCDDataset(f"{cfg['data_dir']}/{cfg['train']}", vocab)
dev_set = ViOCDDataset(f"{cfg['data_dir']}/{cfg['dev']}", vocab)

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
model = TransformerModel(
    vocab_size=vocab.vocab_size,
    num_classes=4,  # 4 domains
    d_model=cfg["d_model"],
    n_heads=cfg["head"],  
    n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    dropout=cfg["dropout"],
    pad_idx=vocab.pad_idx
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer
optimizer = Adam(model.parameters(), lr=cfg["lr"])

# Training function
def train_epoch(epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["domains"].to(device)

        logits, loss = model(input_ids, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    return total_loss / len(train_loader)


# Evaluate
def evaluate(epoch):
    model.eval()
    correct, total = 0, 0

    pbar = tqdm(dev_loader, desc=f"Epoch {epoch} - Evaluating")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["domains"].to(device)

            logits, _ = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

  
# Training loop
best_acc = 0.0

for epoch in range(1, cfg["epochs"] + 1):
    train_loss = train_epoch(epoch)
    acc = evaluate(epoch)

    print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "accuracy": acc
            },
            "checkpoints/transformer_best_classification.pt"
        )
        print(f"Saved best model (acc={acc:.4f})")

print(f"\nTraining done. Best Accuracy = {best_acc:.4f}")

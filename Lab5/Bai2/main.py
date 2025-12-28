import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from phoNER_dataset import load_phoner_data, PhoNERDataset, collate_fn
from sequential_labeling import TransformerModel  

# Load config 
with open("seq_labeling.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("checkpoints", exist_ok=True)

# Load data
train_list, dev_list, test_list = load_phoner_data(
    cfg["data_dir"],
    data_type=cfg.get("data_type", "word")
)

train_dataset = PhoNERDataset(train_list)
dev_dataset   = PhoNERDataset(
    dev_list,
    word2idx=train_dataset.word2idx,
    tag2idx=train_dataset.tag2idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=lambda x: collate_fn(
        x,
        word_pad_id=train_dataset.word2idx["<PAD>"],
        tag_pad_id=train_dataset.tag2idx["<PAD>"],
        max_len=cfg.get("max_len")
    )
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=cfg["batch_size"],
    shuffle=False,
    collate_fn=lambda x: collate_fn(
        x,
        word_pad_id=train_dataset.word2idx["<PAD>"],
        tag_pad_id=train_dataset.tag2idx["<PAD>"],
        max_len=cfg.get("max_len")
    )
)

print(f"Train size: {len(train_dataset)}")
print(f"Dev size: {len(dev_dataset)}")
print(f"Vocab size: {len(train_dataset.word2idx)}")
print(f"Num tags: {len(train_dataset.tag2idx)}")

model = TransformerModel(
    vocab_size=len(train_dataset.word2idx),
    num_tags=len(train_dataset.tag2idx),
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    dropout=cfg["dropout"],
    pad_idx=train_dataset.word2idx["<PAD>"]
).to(device)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = AdamW(model.parameters(), lr=cfg["lr"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2)

# Train one epoch
def train_epoch(epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for words, tags, lengths in pbar:
        words = words.to(device)
        tags = tags.to(device)

        optimizer.zero_grad()
        logits, loss = model(words, tags)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    return total_loss / len(train_loader)

# Evaluate (token accuracy)
def evaluate(epoch):
    model.eval()
    correct, total = 0, 0
    
    pbar = tqdm(dev_loader, desc=f"Epoch {epoch} - Evaluating")
    with torch.no_grad():
        for words, tags, lengths in pbar:
            words = words.to(device)
            tags = tags.to(device)

            logits, _ = model(words, tags)
            preds = logits.argmax(dim=-1)

            mask = tags != train_dataset.tag2idx["<PAD>"]
            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0

# Training loop
best_acc = 0.0
patience_counter = 0
max_patience = 5  

for epoch in range(1, cfg["epochs"] + 1):
    train_loss = train_epoch(epoch)
    dev_acc = evaluate(epoch)

    print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Dev acc: {dev_acc:.4f}")

    scheduler.step(dev_acc)

    if dev_acc > best_acc:
        best_acc = dev_acc
        patience_counter = 0  # Reset patience
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "word2idx": train_dataset.word2idx,
                "tag2idx": train_dataset.tag2idx,
                "dev_acc": dev_acc,
                "config": cfg
            },
            "checkpoints/phoner_best.pt"
        )
        print(f"Saved best model (acc: {best_acc:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{max_patience}")

        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

print(f"Training completed! Best dev accuracy: {best_acc:.4f}")
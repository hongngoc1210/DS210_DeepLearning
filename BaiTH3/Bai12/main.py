import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os

from utils import calculate_f1
from vsfc_dataset import VSFCDataset, load_data, collate_fn, build_vocab, PAD_TOKEN

from lstm import LSTMClassifier
from gru import GRUClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(dataloader, model, criterion, device):
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds.append(outputs)
            trues.append(labels)

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    metrics = calculate_f1(preds, trues)
    metrics['loss'] = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return metrics

def train_loop(train_loader, dev_loader, model, criterion, optimizer, scheduler, num_epochs, device, save_path='best_model.pth'):
    best_score = -1.0
    for epoch in range(num_epochs):
        model.train()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for sequences, labels, lengths in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_preds = torch.cat(all_preds, dim=0)
        train_labels = torch.cat(all_labels, dim=0)
        train_metrics = calculate_f1(train_preds, train_labels)
        print(f"Epoch {epoch+1} TRAIN f1_weighted: {train_metrics['f1_weighted']:.4f} acc: {train_metrics['accuracy']:.4f} avg_loss: {total_loss/len(train_loader):.4f}")

        # validation
        dev_metrics = evaluate(dev_loader, model, criterion, device)
        print(f"Epoch {epoch+1} DEV  f1_weighted: {dev_metrics['f1_weighted']:.4f} acc: {dev_metrics['accuracy']:.4f} loss: {dev_metrics['loss']:.4f}")

        # save best
        if dev_metrics['f1_weighted'] > best_score:
            best_score = dev_metrics['f1_weighted']
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'f1': best_score
            }, save_path)
            print(f"  Saved new best model (f1={best_score:.4f}) -> {save_path}")

        if scheduler:
            scheduler.step()

    return best_score

if __name__ == "__main__":
    torch.manual_seed(42)

    # === CONFIG ===
    train_path = "data/train.json"  # sửa path phù hợp
    dev_path = "data/dev.json"
    test_path = "data/test.json"
    max_len = 100
    batch_size = 16
    num_epochs = 10
    lr = 1e-3
    use_gru = False  # True để dùng GRU
    bidirectional = False

    # === LOAD ===
    train_data, dev_data, test_data, vocab, label_to_idx, idx_to_label = load_data(train_path, dev_path, test_path)

    if len(vocab) <= 2:
        # fallback: build vocab from train_data
        vocab = build_vocab(train_data, min_freq=1, max_vocab_size=30000)

    print(f"Vocab size: {len(vocab)}")
    num_classes = len(label_to_idx)
    print(f"Labels: {label_to_idx}")

    # datasets & dataloaders
    train_dataset = VSFCDataset(train_data, vocab, label_to_idx, max_len=max_len)
    dev_dataset = VSFCDataset(dev_data, vocab, label_to_idx, max_len=max_len)
    test_dataset = VSFCDataset(test_data, vocab, label_to_idx, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=vocab[PAD_TOKEN], max_len=max_len))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id=vocab[PAD_TOKEN], max_len=max_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id=vocab[PAD_TOKEN], max_len=max_len))

    # === MODEL ===
    if use_gru:
        model = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_size=256,
                              num_layers=5, num_classes=num_classes, dropout=0.3,
                              bidirectional=bidirectional, padding_idx=vocab[PAD_TOKEN])
    else:
        model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_size=256,
                               num_layers=5, num_classes=num_classes, dropout=0.5,
                               bidirectional=bidirectional, padding_idx=vocab[PAD_TOKEN])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # train
    best = train_loop(train_loader, dev_loader, model, criterion, optimizer, scheduler, num_epochs, device, save_path='best_model.pth')
    print("Training finished. Best dev f1:", best)

    # load best and evaluate on test
    if os.path.exists('best_model.pth'):
        ckpt = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(ckpt['model_state'])
        test_metrics = evaluate(test_loader, model, criterion, device)
        print("TEST f1_weighted:", test_metrics['f1_weighted'], "acc:", test_metrics['accuracy'], "loss:", test_metrics['loss'])
    else:
        print("No saved model found to evaluate on test.")

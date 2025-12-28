import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from biLSTM import BiLSTMEncoder
from phoNer_dataset import load_phoner_data, PhoNERDataset, collate_fn, PAD_TOKEN, UNK_TOKEN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader, criterion, device, tag_pad_idx):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for words, tags, lengths in tqdm(dataloader, desc='Evaluating'):
            words = words.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)

            outputs = model(words, lengths)  # [B, L, num_tags]
            B, L, C = outputs.size()
            outputs_flat = outputs.view(-1, C)        # [B*L, C]
            tags_flat = tags.view(-1)                 # [B*L]

            loss = criterion(outputs_flat, tags_flat)
            total_loss += loss.item()

            # get predictions per token
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # [B, L]
            gold = tags.cpu().numpy()

            for p_seq, g_seq, ln in zip(preds, gold, lengths.cpu().numpy()):
                p_seq = p_seq[:ln]
                g_seq = g_seq[:ln]
                # remove pad tokens (they shouldn't appear since we truncated by ln)
                all_preds.extend(p_seq.tolist())
                all_labels.extend(g_seq.tolist())

    # remove PAD tag from metrics if tag_pad_idx used in labels (shouldn't be needed because we truncated by lengths)
    # but ensure convert to ints
    all_preds = [int(x) for x in all_preds]
    all_labels = [int(x) for x in all_labels]

    # compute token-level F1 (excluding PAD tag)
    if len(all_labels) == 0:
        return {'loss': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0}

    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    return {'loss': avg_loss, 'f1_macro': f1_macro, 'f1_micro': f1_micro}

def train(train_loader, dev_loader, model, optimizer, scheduler, criterion, device, tag_pad_idx, num_epochs=10, save_path='best_ner.pth'):
    best_dev_f1 = -1.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for words, tags, lengths in pbar:
            words = words.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(words, lengths)  # [B, L, C]
            B, L, C = outputs.size()
            outputs_flat = outputs.view(-1, C)    # [B*L, C]
            tags_flat = tags.view(-1)             # [B*L]

            loss = criterion(outputs_flat, tags_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)
        dev_metrics = evaluate(model, dev_loader, criterion, device, tag_pad_idx)
        print(f"Epoch {epoch} Train loss: {avg_train_loss:.4f} | Dev f1_macro: {dev_metrics['f1_macro']:.4f} f1_micro: {dev_metrics['f1_micro']:.4f} loss: {dev_metrics['loss']:.4f}")

        if dev_metrics['f1_micro'] > best_dev_f1:
            best_dev_f1 = dev_metrics['f1_micro']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'f1_micro': best_dev_f1
            }, save_path)
            print(f"Saved best model (f1_micro={best_dev_f1:.4f}) -> {save_path}")

        if scheduler is not None:
            scheduler.step()

    return best_dev_f1

if __name__ == "__main__":
    torch.manual_seed(42)

    # CONFIG - chỉnh đường dẫn phù hợp
    data_dir = "D:\DS201\DS201.2_TH\BaiTH3\datasets\PhoNER"   # thư mục chứa subfolders 'word' hoặc 'syllable' với train_word.json, ...
    data_type = "word"
    max_len = None   # nếu muốn giới hạn max length, set int
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    num_layers = 5
    embedding_dim = 128
    hidden_size = 256
    dropout = 0.3

    # LOAD raw lists
    train_list, dev_list, test_list = load_phoner_data(data_dir, data_type=data_type)

    # build dataset instances (they will build vocab/tagmap if not provided)
    train_dataset = PhoNERDataset(train_list)
    dev_dataset = PhoNERDataset(dev_list, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)
    test_dataset = PhoNERDataset(test_list, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)

    word2idx = train_dataset.word2idx
    tag2idx = train_dataset.tag2idx
    idx2tag = train_dataset.idx2tag

    print("Vocab size:", len(word2idx), "Num tags:", len(tag2idx))

    word_pad_id = word2idx[PAD_TOKEN]
    tag_pad_id = tag2idx[PAD_TOKEN]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, word_pad_id, tag_pad_id, max_len))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, word_pad_id, tag_pad_id, max_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, word_pad_id, tag_pad_id, max_len))

    # MODEL
    model = BiLSTMEncoder(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_tags=len(tag2idx),
        dropout=dropout,
        padding_idx=word_pad_id
    ).to(device)

    # LOSS (ignore pad tag)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_id)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best = train(train_loader, dev_loader, model, optimizer, scheduler, criterion, device, tag_pad_id, num_epochs=num_epochs, save_path='best_ner.pth')
    print("Best dev f1_micro:", best)

    # load best and evaluate on test
    if os.path.exists('best_ner.pth'):
        ckpt = torch.load('best_ner.pth', map_location=device)
        model.load_state_dict(ckpt['model_state'])
        test_metrics = evaluate(model, test_loader, criterion, device, tag_pad_id)
        print("TEST f1_macro:", test_metrics['f1_macro'], "f1_micro:", test_metrics['f1_micro'], "loss:", test_metrics['loss'])
    else:
        print("No saved model found.")

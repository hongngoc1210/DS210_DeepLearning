import json
import re
from collections import Counter
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def text_to_ids(text, vocab, max_len=100):
    """
    Convert text -> list of ids (variable length, not padded here).
    """
    tokens = text.strip().split()
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids

def build_vocab(data_list, min_freq=1, max_vocab_size=None):
    """
    data_list: list of dicts with 'sentence'
    """
    counter = Counter()
    for item in data_list:
        tokens = clean_text(item['sentence']).split()
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    words = [w for w, f in counter.items() if f >= min_freq]
    words = sorted(words, key=lambda w: counter[w], reverse=True)

    if max_vocab_size:
        words = words[:max_vocab_size - len(vocab)]

    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab

def collate_fn(batch, pad_token_id=0, max_len=100):
    """
    batch is list of tuples: (ids_list (variable len), label_tensor)
    Return: padded_sentences (Tensor [B, L]), labels (Tensor [B]), lengths (Tensor [B])
    """
    sentences = []
    labels = []
    lengths = []
    for item in batch:
        sentence_ids, label = item  # sentence_ids: list or 1D tensor
        if isinstance(sentence_ids, torch.Tensor):
            ids = sentence_ids.tolist()
        else:
            ids = list(sentence_ids)
        if len(ids) > max_len:
            ids = ids[:max_len]
        lengths.append(len(ids))
        sentences.append(torch.tensor(ids, dtype=torch.long))
        labels.append(label)

    padded = pad_sequence(sentences, batch_first=True, padding_value=pad_token_id)  # [B, L_max]
    # If L_max < max_len, pad to max_len; if > max_len already clipped above
    if padded.size(1) < max_len:
        pad_amount = max_len - padded.size(1)
        extra = torch.full((padded.size(0), pad_amount), pad_token_id, dtype=torch.long)
        padded = torch.cat([padded, extra], dim=1)
    elif padded.size(1) > max_len:
        padded = padded[:, :max_len]

    labels = torch.tensor([int(l) if not isinstance(l, torch.Tensor) else int(l.item()) for l in labels], dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, labels, lengths

class VSFCDataset(Dataset):
    def __init__(self, data, vocab, label_to_idx, max_len=100):
        """
        data: list of {'sentence':..., 'sentiment':...}
        vocab: dict token->id
        label_to_idx: dict sentiment->int
        """
        self.data = data
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = clean_text(item['sentence'])
        ids = text_to_ids(sentence, self.vocab, max_len=self.max_len)
        label = self.label_to_idx[item['sentiment']]
        return ids, label

def load_data(train_path, dev_path=None, test_path=None, label_list=None):
    """
    Load json files (list of dicts). If only a single file (folder) provided, you can adapt.
    Returns: train_data, dev_data, test_data
    """
    def _load(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                # fallback: tsv-like file
                data = []
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sent, label = parts[0], parts[1]
                        data.append({'sentence': sent, 'sentiment': label})
            return data

    train = _load(train_path) if train_path else []
    dev = _load(dev_path) if dev_path else []
    test = _load(test_path) if test_path else []

    # build label_to_idx from provided label_list or discovered labels in train/dev/test
    labels = set()
    for d in (train + dev + test):
        labels.add(d['sentiment'])
    if label_list:
        labels = label_list
    labels = sorted(list(labels))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    # build vocab from train only (recommended)
    vocab = build_vocab(train if train else (train + dev + test))
    return train, dev, test, vocab, label_to_idx, idx_to_label

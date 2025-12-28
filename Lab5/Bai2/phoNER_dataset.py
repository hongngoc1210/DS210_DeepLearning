import json
import os
from collections import Counter
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def _read_jsonlines(path: str):
    """
    Expect each line a JSON object like: {"words": [...], "tags": [...]}
    Returns list of dicts.
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(data)
    return items

def load_phoner_data(data_dir: str, data_type: str = "word"):
    """
    data_dir: root folder where there is train_word.json, dev_word.json, test_word.json (or syllable)
    data_type: "word" or "syllable" (depends on your dataset files)
    returns: train_list, dev_list, test_list (each is list of {'words': [...], 'tags': [...]})
    """
    type_dir = os.path.join(data_dir, data_type)
    train_path = os.path.join(type_dir, f"train_{data_type}.json")
    dev_path = os.path.join(type_dir, f"dev_{data_type}.json")
    test_path = os.path.join(type_dir, f"test_{data_type}.json")

    train = _read_jsonlines(train_path)
    dev = _read_jsonlines(dev_path)
    test = _read_jsonlines(test_path)
    return train, dev, test

class PhoNERDataset(Dataset):
    """
    Dataset expects data_list: list of {'words': [...], 'tags': [...]}
    If word2idx / tag2idx not provided, it builds them from data_list.
    """
    def __init__(self, data_list: List[dict], word2idx=None, tag2idx=None):
        super().__init__()
        self.sentences = [d['words'] for d in data_list]
        self.tags = [d['tags'] for d in data_list]

        if word2idx is None:
            self.word2idx = self._build_word_vocab(self.sentences)
        else:
            self.word2idx = word2idx

        if tag2idx is None:
            self.tag2idx = self._build_tag_vocab(self.tags)
        else:
            self.tag2idx = tag2idx

        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def _build_word_vocab(self, sentences, min_freq=1, max_size=None):
        cnt = Counter()
        for s in sentences:
            cnt.update(s)
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        words = [w for w, f in cnt.items() if f >= min_freq]
        words = sorted(words, key=lambda w: cnt[w], reverse=True)
        if max_size:
            words = words[: max_size - len(vocab)]
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        return vocab

    def _build_tag_vocab(self, tags_list):
        tagset = set()
        for seq in tags_list:
            tagset.update(seq)
        tag_vocab = {PAD_TOKEN: 0}
        for t in sorted(tagset):
            if t not in tag_vocab:
                tag_vocab[t] = len(tag_vocab)
        return tag_vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags[idx]
        word_ids = [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in words]
        tag_ids = [self.tag2idx[t] for t in tags]
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], word_pad_id=0, tag_pad_id=0, max_len=None):
    """
    batch: list of (word_ids_tensor, tag_ids_tensor)
    returns: words_padded [B, L], tags_padded [B, L], lengths [B]
    """
    words, tags = zip(*batch)
    lengths = torch.tensor([w.size(0) for w in words], dtype=torch.long)
    if max_len is not None:
        # optionally clip
        words = [w[:max_len] if w.size(0) > max_len else w for w in words]
        tags = [t[:max_len] if t.size(0) > max_len else t for t in tags]
        lengths = torch.tensor([min(l, max_len) for l in lengths], dtype=torch.long)

    words_padded = pad_sequence(words, batch_first=True, padding_value=word_pad_id)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=tag_pad_id)
    return words_padded, tags_padded, lengths

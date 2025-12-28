import json
import os
import re
import torch
from collections import Counter

class Vocab:
    def __init__(self, path: str, min_freq: int = 1):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.pad_idx = 0
        self.unk_idx = 1

        self.build_vocab(path, min_freq)

    def build_vocab(self, path: str, min_freq: int):
        counter = Counter()

        json_files = os.listdir(path)
        for jf in json_files:
            data = json.load(open(os.path.join(path, jf), encoding="utf-8"))
            for item in data.values():
                tokens = self.preprocess(item["review"])
                counter.update(tokens)

        self.itos = [self.pad_token, self.unk_token]
        for w, f in counter.items():
            if f >= min_freq:
                self.itos.append(w)

        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def preprocess(self, text: str):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([!?;,()\"'])", r" \1 ", text)
        text = text.strip()
        return text.split()

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        tokens = self.preprocess(sentence)
        ids = [self.stoi.get(t, self.unk_idx) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    @property
    def vocab_size(self):
        return len(self.itos)

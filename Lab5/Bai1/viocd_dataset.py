import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocab import Vocab

LABEL2ID = {
    "complaint": 1,
    "non-complaint": 0
}


DOMAIN2ID = {
    "fashion": 0,
    "app": 1,
    "cosmetic": 2,
    "mobile": 3
}

def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    labels = torch.tensor([item["label"] for item in items])
    domains = torch.tensor([item["domain"] for item in items])

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0  # pad_idx
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "domains": domains
    }


class ViOCDDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        raw_data = json.load(open(path, encoding="utf-8"))

        # convert dict -> list (giữ thứ tự)
        self.data = [raw_data[k] for k in sorted(raw_data.keys(), key=int)]

        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        review = item["review"]
        label = item["label"]
        domain = item["domain"]

        encoded_review = self.vocab.encode_sentence(review)

        return {
            "input_ids": encoded_review,          # Tensor(L)
            "label": LABEL2ID[label],              # int
            "domain": DOMAIN2ID[domain]             # int
        }

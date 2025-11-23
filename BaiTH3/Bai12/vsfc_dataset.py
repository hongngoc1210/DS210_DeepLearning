import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def collate_fn(batch):
    sentences = [item['sentence'] for item in batch]
    sentiments = [item['sentiment'] for item in batch]
    vocab = build_vocab(batch)
    
    sentence_tensors = [torch.tensor(text_to_ids(s, vocab), dtype=torch.long) for s in sentences]
    padded_sentences = pad_sequence(sentence_tensors, batch_first=True, padding_value=vocab[PAD_TOKEN])
    
    sentiment_tensors = torch.tensor(sentiments, dtype=torch.long)
    
    return padded_sentences, sentiment_tensors
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_vocab(texts, min_freq=2, max_vocab_size=None):
    counter = Counter()
    for t in texts:
        tokens = t['sentence'].lower().split()
        counter.update(tokens)
        
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for word, freq in counter.items():
        if freq >= min_freq:
            if max_vocab_size is None and len(vocab) < max_vocab_size:
                vocab[word] = len(vocab)
            else:
                break
            
    return vocab
    
def text_to_ids(text, vocab, max_len=100):
    tokens = text.split()
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    ids = ids[:max_len]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids
    
    
class VSFCDataset(Dataset):
    def __init__(self, data, vocab, label_to_idx):
        self.data = data
        self.vocab = vocab
        self.label_to_idx = label_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __get_item__(self, idx):
        item = self.data[idx]
        sentence = clean_text(item['sentence'])
        sentence_ids = text_to_ids(sentence, self.vocab)
        sentiment = self.label_to_idx[item['sentiment']]
        return {
            torch.tensor(sentence_ids, dtype=torch.long),
            torch.tensor(sentiment, dtype=torch.long)
        }
    
    def _load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for line in f:
                sentence, sentiment = line.strip().split('\t')
                data.append({'sentence': sentence, 'sentiment': sentiment})
        
        return data

def create_dataloader(train, dev, test, vocab, label_to_idx, batch_size=32):
    train_dataset = VSFCDataset(train, vocab, label_to_idx)
    dev_dataset = VSFCDataset(dev, vocab, label_to_idx)
    test_dataset = VSFCDataset(test, vocab,label_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn= collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size = batch_size,
        shuffle=True,
        collate_fn= collate_fn
    )
    
    return train_loader, test_loader, dev_loader
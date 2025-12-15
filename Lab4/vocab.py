import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
import os
import re
from typing import List, Optional, Union
class Vocab:
    def __init__(self, path: list, src_language: str, tgt_language: str):
        self.initialize_special_tokens()
        self.make_vocab(path, src_language, tgt_language)
        
        self.src_language = src_language
        self.tgt_language = tgt_language
        
    def initialize_special_tokens(self) -> None:
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        
        self.specials = (self.pad_token, self.bos_token, self.eos_token, self.unk_token)
        
        self.bos_idx = 0
        self.eos_idx = 1
        self.pad_idx = 2
        self.unk_idx = 3
    def make_vocab(self, path: str, src_language: str, tgt_language: str):
        
        json_files = os.listdir(path)
        src_word = set()
        tgt_word = set()
        for json_file in json_files:
            data = json.load(open(os.path.join(path, json_file), encoding='utf-8'))
            for item in data:
                src_sentence = item[src_language]
                tgt_sentence = item[tgt_language]
                
                src_tokens = self.preprocesse_sentence(src_sentence)
                tgt_tokens = self.preprocesse_sentence(tgt_sentence)
                
                src_word.update(src_tokens)
                tgt_word.update(tgt_tokens)
                
        src_itos = list(self.specials) + list(src_word)
        self.src_itos = {i: tok for i, tok in enumerate(src_itos)}
        self.src_stoi = {tok: i for i, tok in enumerate(src_itos)}
        
        tgt_itos = list(self.specials) + list(tgt_word)
        self.tgt_itos = {i: tok for i, tok in enumerate(tgt_itos)}
        self.tgt_stoi = {tok: i for i, tok in enumerate(tgt_itos)}
        
    @property   
    def total_src_tokens(self) -> int:
        return len(self.src_itos)

    @property
    def total_tgt_tokens(self)->int:
        return len(self.tgt_itos)
        
    def preprocesse_sentence (self, sentences: str):
        sentence = sentences.lower()
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"\?", " ? ", sentence)
        sentence = re.sub(r";", " ; ", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"\"", " \" ", sentence)
        sentence = re.sub(r"'", " ' ", sentence)
        sentence = re.sub(r"\(", " ( ", sentence)
        sentence = re.sub(r"\)", " ) ", sentence)
        sentence = re.sub(r"\[", " [ ", sentence)
        sentence = re.sub(r"\]", " ] ", sentence)
        sentence = re.sub(r"/", " / ", sentence)
        
        sentence = " ".join(sentence.strip().split())
        tokens = sentence.strip().split()
        
        return tokens
        
    def encode_sentence (self, sentence: str, language: str) -> torch.Tensor:
        # Turn a sentence into a vector of indices and a sentence length
        tokens = self.preprocesse_sentence(sentence)
        stoi = self.src_itos if language == self.src_language else self.tgt_stoi
        vec = [stoi[token] if token in stoi else self.unk_idx for token in tokens]
        vec = [self.bos_idx] + vec + [self.eos_idx]
        vec = torch.Tensor(vec).long()
        
        return vec
    
    def decode_sentence (self, tensor: torch.Tensor, language: str) -> list[str]:
        sentence_ids = tensor.tolist()
        sentences = []
        itos = self.src_itos if language ==self.src_language else self.tgt_itos
        for sentence_ids in sentence_ids:
            words = [itos[idx] for idx in sentence_ids if idx not in self.specials]
            sentence = " ".join(words)
            sentences.append(sentence)
            
        return sentences
        
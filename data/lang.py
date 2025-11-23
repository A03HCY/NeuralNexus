import unicodedata
import re
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Language:
    def __init__(self, name):
        self.name = name
        self.words_index = {}
        self.index_words = {
            0: '<pad>',
            1: '<bos>',
            2: '<eos>',
            3: '<unk>'
        }
        self.n_words = 4

        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
    
    def add_word(self, word):
        if word not in self.words_index:
            self.words_index[word] = self.n_words
            self.index_words[self.n_words] = word
            self.n_words += 1
    
    def add_sentence(self, sentence):
        sentence = normalize_string(sentence)
        for word in sentence.split(' '):
            self.add_word(word)
    
    def get_index(self, word):
        if word in self.words_index:
            return self.words_index[word]
        else:
            return self.words_index['<unk>']
    
    def get_word(self, index):
        if index in self.index_words:
            return self.index_words[index]
        else:
            return '<unk>'
    
    def get_indexes(self, sentence):
        sentence = normalize_string(sentence)
        return [self.get_index(word) for word in sentence.split(' ')]
    
    def get_words(self, indexes):
        return [self.get_word(index) for index in indexes]
    
    def __len__(self):
        return self.n_words


def load_data(file_path, lang_1, lang_2):
    lang_1 = Language(lang_1)
    lang_2 = Language(lang_2)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []

    for line in lines:
        text_1, text_2 = line.replace('\n', '').split('\t')
        lang_1.add_sentence(text_1)
        lang_2.add_sentence(text_2)
        data.append((text_1, text_2))
    
    return lang_1, lang_2, data

class TwoLangDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], src_lang: Language, tgt_lang: Language, add_sign=True):
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.add_sign = add_sign
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_1, text_2 = self.data[idx]
        
        src_indexes = self.src_lang.get_indexes(text_1)
        tgt_indexes = self.tgt_lang.get_indexes(text_2)
        if self.add_sign:
            src_indexes = [self.src_lang.BOS_IDX] + src_indexes + [self.src_lang.EOS_IDX]
            tgt_indexes = [self.tgt_lang.BOS_IDX] + tgt_indexes + [self.tgt_lang.EOS_IDX]
            
        return torch.tensor(src_indexes, dtype=torch.long), torch.tensor(tgt_indexes, dtype=torch.long)

def tld_collate_fn(batch):
    """
    用于 DataLoader 的 collate_fn。
    batch: list of tuples (src_tensor, tgt_tensor)
    """
    src_batch, tgt_batch = [], []
    
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    
    # pad_sequence 会自动把所有 tensor 补齐到最长长度
    # padding_value=0 对应 Language 类中的 <pad>
    # batch_first=True 让输出 shape 为 [Batch, Seq_Len]
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    
    return src_batch, tgt_batch
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from utils.trainer import Trainer

from data.wiki_csai import AnswersDataset
from typing import Union, List, Tuple

batch_size = 64

dataset: Dataset = AnswersDataset('data/wiki_csai.jsonl')

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)
    yield '<pad>'

vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(dataset))

PAD_IDX = vocab['<pad>']
VOCAB_SIZE = len(vocab)

print(f"Vocab size: {VOCAB_SIZE}")

def collate_batch(batch: List[Tuple[str, torch.Tensor]]):
    labels_list, text_list, lengths_list = [], [], [] # 增加 lengths_list
    for (text, label) in batch:
        labels_list.append(label)
        tokens = tokenizer(text)
        processed_text = torch.tensor([vocab[token] for token in tokens], dtype=torch.int64)
        text_list.append(processed_text)
        lengths_list.append(len(processed_text)) # 记录每个句子的真实长度
    labels_tensor = torch.stack(labels_list)
    text_tensor_padded = pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    
    return text_tensor_padded, labels_tensor

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super(LSTMClassifier, self).__init__()
        # 增加 padding_idx 参数，让模型在计算embedding时忽略填充符
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                              hidden_dim, 
                              num_layers=2,          # 增加层数
                              bidirectional=True,    # 使用双向
                              dropout=0.5,           # 添加Dropout
                              batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)
    
    def forward(self, text_batch: torch.Tensor) -> torch.Tensor:
        # text_batch: [batch_size, seq_len]
        embedded = self.embedding(text_batch.long()) # [batch_size, seq_len, embedding_dim]

        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        fc_out = self.fc(final_hidden_state)
        return fc_out

model = LSTMClassifier(VOCAB_SIZE, embedding_dim=128, hidden_dim=128, pad_idx=PAD_IDX)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_trainer = Trainer(
    model,
    10,
    train_loader=dataloader,
    optimizer=optimizer,
    criterion=criterion,
    checkpoint_path='checkpoints/lstm_class_model.pth'
)

for trainer in model_trainer.train(tqdm_bar=True, print_loss=True):
    trainer.auto_update()
    trainer.auto_checkpoint()

correct = 0
total = 0
for trainer in model_trainer.eval(dataloader, tqdm_bar=True):
    logist = model.forward(trainer.data)
    _, predicted = torch.max(logist.data, 1)
    _, cor = torch.max(trainer.target.data, 1)
    total += trainer.target.size(0)
    correct += (predicted == cor).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

model_trainer.save_model('models/lstm_class_model.pth')
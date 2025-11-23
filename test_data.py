from data.lang import *
from torch.utils.data import DataLoader

eng, fra, data = load_data('./data/eng_fra.txt', 'eng', 'fra')

dataset = TwoLangDataset(data, eng, fra)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tld_collate_fn)

for src, tgt in dataloader:
    print("Source shape:", src.shape) # [32, max_len_src]
    print("Target shape:", tgt.shape) # [32, max_len_tgt]
    
import torch
import torch.nn as nn
import time
from utils.block import Transformer
from data.lang import load_data, TwoLangDataset, tld_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """
    Args:
        model: 训练好的 Transformer 模型
        src: 源句子 Tensor [1, Src_Len]
        src_mask: 源句子掩码 [1, 1, 1, Src_Len]
        max_len: 最大生成长度
        start_symbol: <BOS> 的索引
        end_symbol: <EOS> 的索引
    """
    model.eval() # 必须切换到评估模式
    
    # 1. 编码器只需运行一次，得到记忆 enc_output
    with torch.no_grad():
        enc_output = model.encoder_forward(src, src_mask)
    
    # 2. 初始化解码器输入：一开始只有一个 <BOS>
    dec_input = torch.ones(1, 1).fill_(start_symbol).type_as(src).long()
    
    # 3. 循环生成
    for i in range(max_len - 1):
        # 生成解码器掩码
        dec_mask = model.create_tgt_mask(dec_input) # [1, 1, L, L]
        
        with torch.no_grad():
            # 将当前的输入和记忆喂给解码器
            out = model.decoder_forward(dec_input, enc_output, src_mask, dec_mask)
            
            # 经过线性层映射到词表
            prob = model.ffn(out) # [1, seq_len, vocab_size]
        
        # 取最后一个时间步的输出 (即预测的下一个词)
        # prob[:, -1, :] shape: [1, vocab_size]
        _, next_word = torch.max(prob[:, -1, :], dim=1)
        next_word_item = next_word.item()
        
        # 将预测的词拼接到输入序列中，用于下一次循环
        dec_input = torch.cat([dec_input, torch.ones(1, 1).type_as(src).fill_(next_word_item)], dim=1)
        
        # 如果预测到了结束符 <EOS>，停止生成
        if next_word_item == end_symbol:
            break
            
    return dec_input

def translate(sentence, model, src_lang, tgt_lang, device, max_len=50):
    model.eval()
    
    # 1. 文本预处理：分词 -> 索引转换
    src_indexes = src_lang.get_indexes(sentence)
    
    # 2. 添加特殊符号 (Encoder 最好加上 BOS/EOS 以匹配训练时的分布)
    src_indexes = [src_lang.BOS_IDX] + src_indexes + [src_lang.EOS_IDX]
    
    # 3. 转为 Tensor 并增加 Batch 维度 [Seq_Len] -> [1, Seq_Len]
    src_tensor = torch.tensor(src_indexes).unsqueeze(0).to(device)
    
    # 4. 创建掩码
    src_mask = model.create_src_mask(src_tensor, pad_idx=src_lang.PAD_IDX)
    
    # 5. 贪婪解码
    tgt_tokens = greedy_decode(
        model, 
        src_tensor, 
        src_mask, 
        max_len=max_len, 
        start_symbol=tgt_lang.BOS_IDX, 
        end_symbol=tgt_lang.EOS_IDX,
        device=device
    )
    
    # 6. 索引 -> 文本转换
    # tgt_tokens 是 Tensor [1, Len]，先转 list，去掉 batch 维度
    tgt_indexes = tgt_tokens.squeeze(0).tolist()
    
    # 过滤掉 BOS 和 EOS
    result_words = []
    for idx in tgt_indexes:
        if idx == tgt_lang.BOS_IDX: continue
        if idx == tgt_lang.EOS_IDX: break
        result_words.append(tgt_lang.get_word(idx))
        
    return " ".join(result_words)

# 1. 数据准备
print("Loading data...")
eng, fra, data = load_data('./data/eng_fra.txt', 'eng', 'fra')

print(f"Data loaded. English vocab: {len(eng)}, French vocab: {len(fra)}, Samples: {len(data)}")

dataset = TwoLangDataset(data, eng, fra)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tld_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 模型初始化
model = Transformer(
    model_dim=512,
    head_num=8,      # 建议设为 8，512能被8整除
    layer_num=3,     # 只有3层，训练比较快
    ff_dim=2048,     # 通常是 model_dim * 4
    enc_vocab_size=len(eng),
    dec_vocab_size=len(fra),
    dropout=0.1
).to(device)

# 初始化权重
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

print(model)

# 3. 优化器与损失函数
PAD_IDX = eng.PAD_IDX 

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

test_sentences = [
    "I love cats.",
    "He is a good student.",
    "Where are you going?"
]

# 4. 训练循环
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train() # 开启训练模式
    start_time = time.time()
    epoch_loss = 0
    
    # 使用 tqdm 显示进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    
    for src, tgt in progress_bar:
        # src: [Batch, Src_Len]
        # tgt: [Batch, Tgt_Len]
        pass
        src = src.to(device)
        tgt = tgt.to(device)

        # 构建 Decoder 输入与标签 (Teacher Forcing)
        # [BOS, A, B, C, EOS] -> [BOS, A, B, C]
        dec_input = tgt[:, :-1]
        
        # Label (Target): 去掉第一个 token (BOS)
        # [BOS, A, B, C, EOS] -> [A, B, C, EOS]
        dec_target = tgt[:, 1:]

        # 前向传播
        optimizer.zero_grad()
        
        # outputs: [Batch, Seq_Len-1, Dec_Vocab_Size]
        outputs = model(src, dec_input)

        # 计算损失
        # CrossEntropyLoss 需要:
        # input: [N, C]  -> 将输出展平: [(Batch * Seq_Len), Vocab_Size]
        # target: [N]    -> 将标签展平: [(Batch * Seq_Len)]
        loss = criterion(outputs.reshape(-1, len(fra)), dec_target.reshape(-1))
        
        # 反向传播与更新
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # 更新进度条显示的 Loss
        progress_bar.set_postfix({'loss': loss.item()})

    # Epoch 结束统计
    avg_loss = epoch_loss / len(dataloader)
    end_time = time.time()

    
    print("-" * 30)
    model.eval() # 切换到评估模式 (关闭 Dropout)
    for s in test_sentences:
        trans_res = translate(s, model, eng, fra, device)
        print(f" Src: {s}")
        print(f" Pred: {trans_res}")
    print("-" * 30)
    
    print(f"Epoch: {epoch+1} | Time: {end_time - start_time:.2f}s | Train Loss: {avg_loss:.4f}")
    
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'transformer_epoch_{epoch+1}.pth')

print("Training Finished!")

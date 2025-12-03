import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calculate_causal_layer(step:int, kernel_size:int=3):
    if kernel_size <= 1:
        raise ValueError("kernel_size must be greater than 1")
    L = math.ceil(math.log2((step - 1) / (kernel_size - 1) + 1))
    R = 1 + (kernel_size - 1) * (2 ** L - 1)
    return int(L), R

def apply_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    ''' 应用注意力机制 (Scaled Dot-Product Attention)。
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: 查询张量。Shape: [... , L_q, D]
        K: 键张量。  Shape: [... , L_k, D]
        V: 值张量。  Shape: [... , L_k, D] (通常 K 和 V 的长度是一样的)
        mask: 掩码。 Shape: [B, 1, 1, L_k] 或 [B, 1, L_q, L_k] (用于广播)
    '''
    
    # 1. 获取缩放因子
    d_k = math.sqrt(Q.size(-1)) 
    # 2. 计算注意力分数 (QK^T)
    # Q: [..., L_q, D]
    # K: [..., L_k, D]
    # K.transpose(-2, -1): 交换最后两个维度 -> [..., D, L_k]
    # torch.matmul 会自动识别最后两个维度做矩阵乘法，前面的维度 (...) 视为 Batch 处理
    # 运算: [..., L_q, D] @ [..., D, L_k] -> D 被消掉
    q_k = torch.matmul(Q, K.transpose(-2, -1))
    # q_k Shape: [..., L_q, L_k]  <-- 注意力分数矩阵
    # 3. 缩放 (Scaling)
    # Shape 不变: [..., L_q, L_k]
    scores = q_k / d_k
    # 4. 掩码 (Masking)
    if mask is not None:
        # mask == 0 的位置通常表示 Padding 或者未来的词
        # 将这些位置填入 -1e9 (负无穷)，这样 softmax 后概率会趋近于 0
        # mask 的形状必须能广播到 scores 的形状
        scores = scores.masked_fill(mask == 0, -1e9)
    # 5. 归一化 (Softmax)
    # dim=-1 表示在最后一个维度 (L_k) 上进行 softmax
    # 在最后一个维度 (L_k) 上归一化，表示 Query 对所有 Key 的关注度总和为 1
    # attn Shape: [..., L_q, L_k]
    attn = torch.softmax(scores, dim=-1)
    # 6. 加权求和 (Attention * V)
    # attn: [..., L_q, L_k]
    # V:    [..., L_k, D]
    # 运算: [..., L_q, L_k] @ [..., L_k, D] -> L_k 被消掉
    # 返回 Shape: [..., L_q, D] 
    # 保持与输入 Q 的形状一致
    output = torch.matmul(attn, V)
    return output

def create_src_mask(src: torch.Tensor, pad_idx:int=0) -> torch.Tensor:
    ''' 创建源序列掩码。
    
    Args:
        src: 源序列张量。形状为 [B, L_src]
        pad_idx: 填充符号的索引。默认为 0
    
    Returns:
        mask: 源序列掩码。形状为 [B, 1, 1, L_src]
    '''
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_tgt_mask(tgt: torch.Tensor, pad_idx:int=0) -> torch.Tensor:
    ''' 创建目标序列掩码。
    
    Args:
        tgt: 目标序列张量。形状为 [B, L_tgt]
        pad_idx: 填充符号的索引。默认为 0
        
    Returns:
        mask: 目标序列掩码。形状为 [B, 1, L_tgt, L_tgt]
    '''
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    lookahead_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()

    mask = pad_mask & lookahead_mask
    return mask

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000):
        """
        d_model: 必须是偶数。如果是多头注意力，这里传入的应该是 head_dim。
        """
        super(RotaryPositionalEmbedding, self).__init__()
        
        # 1. 频率计算 (Inv Freq)
        # 维度: [d_model / 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # 2. 生成位置索引 [max_len]
        t = torch.arange(max_len, dtype=torch.float)
        
        # 3. 计算外积得到角度 [max_len, d_model/2]
        freqs = torch.outer(t, inv_freq)
        
        # 4. 拼接频率以匹配 d_model [max_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 5. 预计算 Cos 和 Sin [max_len, d_model]
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x):
        """
        将向量分为两半并旋转: [-x2, x1]
        无论输入是 3D 还是 4D，Split 都是作用在最后一维 (d_model)
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自动适配两种输入:
        1. [Batch, Seq_Len, Dim]
        2. [Batch, Head, Seq_Len, Head_Dim]
        """
        # 获取输入维度信息
        ndim = x.ndim
        seq_len = x.shape[-2] # 无论是 3D(idx 1) 还是 4D(idx 2)，序列长度都在倒数第二维
        
        # 6. 切片获取当前长度对应的 cos/sin
        # [seq_len, d_model]
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # 7. 根据输入维度调整 cos/sin 的形状以支持广播
        if ndim == 4:
            # 输入: [B, H, L, D] -> 需要 cos: [1, 1, L, D]
            cos = cos.view(1, 1, seq_len, -1)
            sin = sin.view(1, 1, seq_len, -1)
        elif ndim == 3:
            # 输入: [B, L, D] -> 需要 cos: [1, L, D]
            cos = cos.view(1, seq_len, -1)
            sin = sin.view(1, seq_len, -1)
        else:
            raise ValueError(f"不支持的输入维度: {ndim}，仅支持 3D 或 4D 输入")

        # 8. 应用旋转
        return (x * cos) + (self._rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num: int, model_dim: int, rope: bool = False, max_len: int = 5000):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % head_num == 0
        self.head_num = head_num
        self.model_dim = model_dim
        self.head_dim = model_dim // head_num
        
        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(model_dim, model_dim)
        self.W_V = nn.Linear(model_dim, model_dim)
        self.W_O = nn.Linear(model_dim, model_dim)
        
        self.use_rope = rope
        if self.use_rope:
            self.rope_layer = RotaryPositionalEmbedding(d_model=self.head_dim, max_len=max_len)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性变换 + 分头
        # [B, L, D] -> [B, L, Head, Head_Dim] -> [B, Head, L, Head_Dim]
        Q = self.W_Q(Q).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        
        if self.use_rope:
            Q = self.rope_layer(Q)
            K = self.rope_layer(K)
        
        # 2. 计算 Attention
        output = apply_attention(Q, K, V, mask)
        
        # 3. 合并头
        # [B, Head, L, Head_Dim] -> [B, L, Head, Head_Dim] -> [B, L, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        
        output = self.W_O(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 1. 初始化矩阵: [max_len, d_model] (例如 [5000, 512])
        pe = torch.zeros(max_len, d_model)
        
        # 2. 生成位置索引向量: [0, 1, ... 4999] -> [5000, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 计算分母项 div_term (控制频率)
        # 这是一个从 1 到 10000 的几何级数衰减
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. 填充矩阵
        # 偶数位(0, 2, 4...)填 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位(1, 3, 5...)填 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. 增加 Batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # register_buffer 意味着这不是可训练参数，但会随模型 state_dict 保存
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: [Batch_Size, Seq_Len, d_model]
        """
        # 6. 切片与相加
        # self.pe 是 [1, 5000, 512]
        # x.size(1) 是当前句子的实际长度 (比如 10)
        # 取出前 10 个位置编码: [1, 10, 512]
        # 利用广播机制加到 x 上
        x = x + self.pe[:, :x.size(1), :]
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim):
        super(PositionwiseFeedForward, self).__init__()
        # model_dim 通常是 512
        # ff_dim (中间层维度) 通常是 2048 (4倍大小)
        
        # 第一层: 扩维 [512 -> 2048]
        self.fc1 = nn.Linear(model_dim, ff_dim)
        
        # 第二层: 降维 [2048 -> 512]
        self.fc2 = nn.Linear(ff_dim, model_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的 shape: [Batch_Size, Seq_Len, d_model]
        
        # 1. 线性投影 1
        # PyTorch 的 Linear 层只会作用于最后一维
        # [B, S, 512] -> [B, S, 2048]
        x = self.fc1(x)
        
        # 2. 激活函数
        # [B, S, 2048] (负数变0)
        x = self.relu(x)
        
        # 3. 线性投影 2
        # [B, S, 2048] -> [B, S, 512]
        x = self.fc2(x)
        return x

class ProjectionHead(nn.Module):
    ''' 通用投影头/前馈网络模块。
    
    支持两种模式：
    1. MLP模式 (use_mlp=True 或 as_ffn=True): Linear -> Act -> Dropout -> Linear
    2. 线性模式 (use_mlp=False 且 as_ffn=False): Linear

    Args:
        in_dim (int): 输入特征维度。
        out_dim (int): 输出特征维度。
        hidden_dim (int, optional): 隐藏层维度。仅在 MLP 模式下有效。
            默认为 None。
            - 如果 as_ffn=True 且 hidden_dim=None，则默认为 in_dim * 4。
            - 如果 use_mlp=True 且 hidden_dim=None，则默认为 in_dim。
        dropout (float): Dropout 概率。默认为 0.0。
        use_mlp (bool): 是否使用 MLP 结构。默认为 False。
        as_ffn (bool): 是否作为标准 FFN 使用。如果为 True，强制 use_mlp=True 且默认 hidden_dim=in_dim*4。默认为 False。
    '''
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None, dropout: float = 0.0, use_mlp: bool = False, as_ffn: bool = False):
        super(ProjectionHead, self).__init__()
        
        if as_ffn:
            use_mlp = True
            if hidden_dim is None:
                hidden_dim = in_dim * 4
        
        layers = []
        
        if use_mlp:
            h_dim = hidden_dim if hidden_dim is not None else in_dim
            
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(h_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。Shape: [..., in_dim]

        Returns:
            torch.Tensor: 输出张量。Shape: [..., out_dim]
        '''
        return self.net(x)

class AIMEncoder(nn.Module):
    def __init__(self, model_dim, head_num, dropout=0.1, max_len=5000):
        super(AIMEncoder, self).__init__()

        self.self_attn = MultiHeadAttention(head_num, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)

        self.cross_attn = MultiHeadAttention(head_num, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.ffn = ProjectionHead(model_dim, model_dim, as_ffn=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(model_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, memory, x, mask=None):
        _memory = self.norm1(memory)
        attn = self.self_attn(_memory, _memory, _memory)
        memory = memory + self.dropout(attn)

        _memory = self.norm2(memory)
        cross_attn = self.cross_attn(_memory, x, x, mask=mask)
        memory = memory + self.dropout(cross_attn)

        _memory = self.norm3(memory)
        ffn = self.ffn(_memory)
        memory = memory + self.dropout(ffn)
        return memory

class AIMDecoder(nn.Module):
    def __init__(self, model_dim, head_num, dropout=0.1, max_len=5000):
        super(AIMDecoder, self).__init__()

        self.self_attn = MultiHeadAttention(head_num, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)

        self.cross_attn = MultiHeadAttention(head_num, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.ffn = ProjectionHead(model_dim, model_dim, as_ffn=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(model_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, dec_mask=None):
        _x = self.norm1(x)
        attn = self.self_attn(_x, _x, _x, dec_mask)
        x = x + self.dropout(attn)

        _x = self.norm2(x)
        cross_attn = self.cross_attn(_x, memory, memory)
        x = x + self.dropout(cross_attn)

        _x = self.norm3(x)
        ffn = self.ffn(_x)
        x = x + self.dropout(ffn)
        return x

class AIM(nn.Module):
    def __init__(self, model_dim, vocab_size, head_num, mem_dim=512, encoder_num=3, decoder_num=3, max_len=5000, dropout=0.1):
        super(AIM, self).__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        
        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, model_dim)

        self.pos_encoding = PositionalEncoding(model_dim, max_len)
        
        # AIM Encoder：用于 Memory 读取 Context
        self.encoder = nn.ModuleList([
            AIMEncoder(model_dim, head_num, dropout, max_len) 
            for _ in range(encoder_num)
        ])
        self.encoder_norm = nn.LayerNorm(model_dim)
        # AIM Decoder：用于生成
        self.decoder = nn.ModuleList([
            AIMDecoder(model_dim, head_num, dropout, max_len) 
            for _ in range(decoder_num)
        ])
        self.decoder_norm = nn.LayerNorm(model_dim)
        self.projection = ProjectionHead(model_dim, vocab_size)
        self.init_memory = nn.Parameter(torch.randn(1, mem_dim, model_dim) * 0.02)

    def get_init_memory(self, batch_size):
        return self.init_memory.expand(batch_size, -1, -1)
    
    def encoder_forward(self, x, memory, detach=True, pad_id=0):
        mask = create_src_mask(x, pad_id)
        x = self.embedding(x) * math.sqrt(self.model_dim)
        x = self.pos_encoding(x)
        
        for encoder in self.encoder:
            memory = encoder(memory, x, mask)
        
        memory = self.encoder_norm(memory)
        
        if detach:
            memory = memory.detach()
        
        return memory
    
    def decoder_forward(self, x, memory, pad_id=0):
        mask = create_tgt_mask(x, pad_id)
        x = self.embedding(x) * math.sqrt(self.model_dim)
        for decoder in self.decoder:
            x = decoder(x, memory, mask)
        
        x = self.decoder_norm(x)
        return x
    
    def forward(self, x, memory=None):
        if memory is None:
            memory = self.get_init_memory(x.size(0))
        x = self.decoder_forward(x, memory)
        x = self.projection(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, head_num, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.self_attn = MultiHeadAttention(head_num, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)

        self.pos_ffn = PositionwiseFeedForward(model_dim, ff_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn = self.self_attn(x, x, x, mask)
        attn = self.dropout(attn)
        x = self.norm1(x + attn)

        ffn = self.pos_ffn(x)
        ffn = self.dropout(ffn)
        x = self.norm2(x + ffn)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, head_num, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.self_attn = MultiHeadAttention(head_num, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)

        self.cross_attn = MultiHeadAttention(head_num, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.pos_ffn = PositionwiseFeedForward(model_dim, ff_dim)
        self.norm3 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, enc_mask, dec_mask):
        attn = self.self_attn(x, x, x, dec_mask)
        attn = self.dropout(attn)
        x = self.norm1(attn + x)

        cros = self.cross_attn(x, enc_output, enc_output, enc_mask)
        cros = self.dropout(cros)
        x = self.norm2(cros + x)

        ffn = self.pos_ffn(x)
        ffn = self.dropout(ffn)
        x = self.norm3(ffn + x)
        return x

class Transformer(nn.Module):
    def __init__(self, model_dim, head_num, layer_num, ff_dim, enc_vocab_size, dec_vocab_size, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.head_num = head_num

        self.encoder_embedding = nn.Embedding(enc_vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(dec_vocab_size, model_dim)

        self.pe = PositionalEncoding(model_dim, max_len)

        self.encoder = nn.ModuleList([TransformerEncoder(model_dim, head_num, ff_dim, dropout) for _ in range(layer_num)])
        self.decoder = nn.ModuleList([TransformerDecoder(model_dim, head_num, ff_dim, dropout) for _ in range(layer_num)])

        self.ffn = nn.Linear(model_dim, dec_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encoder_forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.encoder_embedding(x) * math.sqrt(self.model_dim)
        x = self.pe(x)

        for layer in self.encoder:
            x = layer(x, mask)
        
        return x # [B, seq_len, model_dim]
    
    def decoder_forward(self, x: torch.Tensor, enc_output: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embedding(x) * math.sqrt(self.model_dim)
        x = self.pe(x)

        for layer in self.decoder:
            x = layer(x, enc_output, enc_mask, dec_mask)
        
        return x # [B, seq_len, model_dim]
    
    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
        src_mask = create_src_mask(enc_input)
        tgt_mask = create_tgt_mask(dec_input)

        # enc_input: [B, seq_len]
        # dec_input: [B, seq_len]
        enc_output = self.encoder_forward(enc_input, src_mask)
        dec_output = self.decoder_forward(dec_input, enc_output, src_mask, tgt_mask)

        logits = self.ffn(dec_output)
        return logits # [B, seq_len, dec_vocab_size]

class ConvBlock(nn.Module):
    ''' 带有归一化和激活函数的基本卷积块。
    '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, leaky_relu: float = 0.1, act: bool = True):
        ''' 初始化 ConvBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 卷积步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act (bool): 是否使用激活函数。默认为 True。
        '''
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True) if act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class DoubleConvBlock(nn.Module):
    ''' 由两个 ConvBlock 组成的双卷积块。
    '''
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, kernel_size: int = 3, padding: int = 1, stride: int = 1, leaky_relu: float = 0.1, act_1: bool = True, act_2: bool = True):
        ''' 初始化 DoubleConvBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            mid_ch (int, optional): 中间通道数。默认为 None（等于 out_ch）。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 第一个卷积层的步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act_1 (bool): 第一个卷积层是否使用激活函数。默认为 True。
            act_2 (bool): 第二个卷积层是否使用激活函数。默认为 True。
        '''
        super(DoubleConvBlock, self).__init__()
        mid_ch = mid_ch if mid_ch is not None else out_ch
        self.conv_1 = ConvBlock(in_ch, mid_ch, kernel_size=kernel_size, padding=padding, stride=stride, leaky_relu=leaky_relu, act=act_1)
        self.conv_2 = ConvBlock(mid_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=1, leaky_relu=leaky_relu, act=act_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class DownsampleBlock(nn.Module):
    ''' 使用带步长的 ConvBlock 或 DoubleConvBlock 进行下采样块。
    '''
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, stride: int = 2, leaky_relu: float = 0.1, act: bool = True, use_double_conv: bool = True, maxpool: bool = True, dropout_prob: float = 0.0, return_features: bool = True):
        ''' 初始化 DownsampleBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            kernel_size (int): 卷积核大小。默认为 3。
            padding (int): 输入两侧的零填充。默认为 1。
            stride (int): 卷积步长。默认为 2。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            act (bool): 是否使用激活函数。默认为 True。
            use_double_conv (bool): 是否使用双卷积块。默认为 True。
            maxpool (bool): 是否使用 MaxPool2d(2) 进行下采样。默认为 True。
            dropout_prob (float): Dropout 概率。默认为 0.0。
            return_features (bool): 是否返回中间特征（池化前）。默认为 True。
        '''
        super(DownsampleBlock, self).__init__()
        self.return_features = return_features
        
        conv_stride = 1 if maxpool else stride
        
        if use_double_conv:
            self.block = DoubleConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding, leaky_relu=leaky_relu, act_1=True, act_2=act)
        else:
            self.block = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding, leaky_relu=leaky_relu, act=act)
            
        self.maxpool = nn.MaxPool2d(2) if maxpool else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.block(x)
        features = x
        if self.maxpool is not None:
            x = self.maxpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.return_features:
            return x, features
        return x

class UpsampleBlock(nn.Module):
    ''' 使用转置卷积后接 ConvBlock 或 DoubleConvBlock 的上采样块。
    '''
    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2, leaky_relu: float = 0.1, use_double_conv: bool = True, use_skip: bool = True):
        ''' 初始化 UpsampleBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            scale_factor (int): 空间大小的乘数。默认为 2。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
            use_double_conv (bool): 是否使用双卷积块。默认为 True。
            use_skip (bool): 是否使用跳跃连接。默认为 True。
        '''
        super(UpsampleBlock, self).__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor)
        
        conv_in_ch = out_ch * 2 if use_skip else out_ch
        
        if use_double_conv:
            self.conv = DoubleConvBlock(conv_in_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu)
        else:
            self.conv = ConvBlock(conv_in_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            skip (torch.Tensor, optional): 跳跃连接张量。默认为 None。

        Returns:
            torch.Tensor: 输出张量。
        '''
        x = self.up(x)
        
        if self.use_skip:
            if skip is not None:
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            else:
                raise ValueError("UpsampleBlock expects a skip connection tensor when use_skip=True, but got None.")

        x = self.conv(x)
        return x

class ResBasicBlock(nn.Module):
    ''' 基本残差块。
    '''
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, leaky_relu: float = 0.1):
        ''' 初始化 ResBasicBlock。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            stride (int): 卷积步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
        '''
        super(ResBasicBlock, self).__init__()
        self.conv_1 = ConvBlock(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, leaky_relu=leaky_relu)
        self.conv_2 = ConvBlock(out_ch, out_ch, kernel_size=3, padding=1, stride=1, leaky_relu=leaky_relu, act=False)
        
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ConvBlock(in_ch, out_ch, kernel_size=1, padding=0, stride=stride, act=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        residual = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = self.relu(x)
        return x


class ResBottleneckBlock(nn.Module):
    ''' 残差瓶颈块。
    '''
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, leaky_relu: float = 0.1):
        ''' 初始化 ResBottleneckBlock。

        Args:
            in_ch (int): 输入通道数。
            mid_ch (int): 中间通道数。
            out_ch (int): 输出通道数。
            stride (int): 卷积步长。默认为 1。
            leaky_relu (float): LeakyReLU 的负斜率。默认为 0.1。
        '''
        super(ResBottleneckBlock, self).__init__()
        self.conv_1 = ConvBlock(in_ch, mid_ch, kernel_size=1, padding=0, leaky_relu=leaky_relu)
        self.conv_2 = ConvBlock(mid_ch, mid_ch, kernel_size=3, padding=1, stride=stride, leaky_relu=leaky_relu)

        self.conv_3 = ConvBlock(mid_ch, out_ch, kernel_size=1, padding=0, act=False)
        
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ConvBlock(in_ch, out_ch, kernel_size=1, padding=0, stride=stride, act=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        residual = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = self.relu(x)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, leaky_relu=0.1, use_res=True, dropout=0.2):
        super(CausalConv1d, self).__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.use_res = use_res
        
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation))
        
        self.relu = nn.LeakyReLU(leaky_relu, inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = None
        if use_res and in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, padding=0)

    def forward(self, x):
        residual = x
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.use_res:
            if self.downsample is not None:
                residual = self.downsample(residual)
            
            x = x + residual
            
        return x
    
    @staticmethod
    def auto_block(in_channels, out_channels, step, kernel_size=3, leaky_relu=0.1, use_res=True, dropout=0.2) -> nn.Sequential:
        layers, _ = calculate_causal_layer(step, kernel_size)
        model = []
        for i in range(layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels

            model.append(CausalConv1d(in_ch, out_channels, kernel_size, dilation, leaky_relu, use_res, dropout))

        return nn.Sequential(*model)

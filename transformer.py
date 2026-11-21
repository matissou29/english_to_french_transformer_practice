import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(dim))    # gamma
        self.b = nn.Parameter(torch.zeros(dim))   # beta

    def forward(self, x):
        m = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - m) / torch.sqrt(var + self.eps)
        return self.a * x + self.b

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = nn.Linear(cfg["d_model"], cfg["d_model"] * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(cfg["d_model"] * 4, cfg["d_model"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_head"] == 0, "the n_head must eb a dividor of d_model"

        self.d_in = cfg["d_model"]
        self.d_out = cfg["d_model"]
        self.n_head = cfg["n_head"]
        self.d_head = cfg["d_model"] // cfg["n_head"]

        self.W_q = nn.Linear(cfg["d_model"], cfg["d_model"], bias = cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["d_model"], cfg["d_model"], bias = cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["d_model"], cfg["d_model"], bias = cfg["qkv_bias"])
        self.W_o = nn.Linear(cfg["d_model"], cfg["d_model"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, tgt_mask):

        batch, seq_size, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch, seq_size, self.n_head, self.d_head)
        K = K.view(batch, seq_size, self.n_head, self.d_head)
        V = V.view(batch, seq_size, self.n_head, self.d_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attention_weights = Q @ K.transpose(2,3)
        attention_weights = attention_weights.masked_fill(~tgt_mask, -torch.inf)
        attention_weights = torch.softmax(attention_weights / math.sqrt(self.d_head), dim=-1)


        attention_weights = self.dropout(attention_weights)
        attention_score = (attention_weights @ V).transpose(1,2)

        attention_score = attention_score.contiguous().view(batch, seq_size, d_model)

        return self.W_o(attention_score)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_head"] == 0, "the number of head must be a dividor of embedding size"

        self.d_head = cfg["d_model"] // cfg["n_head"]
        self.n_head = cfg["n_head"]
        self.d_in = cfg["d_model"]
        self.d_out = cfg["d_model"]

        self.dropout = nn.Dropout(cfg["dropout"])

        self.W_q = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_o = nn.Linear(cfg["d_model"], cfg["d_model"])

    def forward(self, x, src_mask):
        batch, seq_size, d_model = x.shape

        # dim: (b, seq_size, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # dim: (b, n_head, seq_size, d_head)
        Q = Q.view(batch, seq_size, self.n_head, self.d_head).transpose(1,2)
        K = K.view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_size, self.n_head, self.d_head).transpose(1,2)

        K = K.transpose(2, 3)

        att_weights= Q @ K
        att_weights = att_weights.masked_fill_(~src_mask, -torch.inf)
        att_weights = torch.softmax(att_weights / math.sqrt(self.d_head), dim=-1)

        att_weights = self.dropout(att_weights)

        att_score = (att_weights @ V).transpose(1, 2)
        att_score = att_score.contiguous().view(batch, seq_size, d_model)

        return self.W_o(att_score)






class MultiHeadCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_head"] == 0, "the number of head must be a dividor of embedding size"

        self.d_head = cfg["d_model"] // cfg["n_head"]
        self.n_head = cfg["n_head"]
        self.d_in = cfg["d_model"]
        self.d_out = cfg["d_model"]

        self.dropout = nn.Dropout(cfg["dropout"])

        self.W_q = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_o = nn.Linear(cfg["d_model"], cfg["d_model"])


    def forward(self, x, memory, src_mask):
        batch, seq_size, d_model = x.shape
        _, seq_size_memory, _ = memory.shape

        # dim: (b, seq_size, d_model)
        Q = self.W_q(x)
        K = self.W_k(memory)
        V = self.W_v(memory)

        # dim: (b, n_head, seq_size, d_head)
        Q = Q.view(batch, seq_size, self.n_head, self.d_head).transpose(1,2)
        K = K.view(batch, seq_size_memory, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_size_memory, self.n_head, self.d_head).transpose(1,2)

        K = K.transpose(2, 3)

        att_weights = Q @ K
        att_weights = att_weights.masked_fill(~src_mask, -torch.inf)
        att_weights = torch.softmax(att_weights / math.sqrt(self.d_head), dim=-1)

        att_weights = self.dropout(att_weights)

        att_score = (att_weights @ V).transpose(1, 2)
        att_score = att_score.contiguous().view(batch, seq_size, d_model)

        return self.W_o(att_score)

class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.multi_head_attention = MultiHeadAttention(cfg)
        self.norm1 = LayerNorm(cfg["d_model"])
        self.norm2 = LayerNorm(cfg["d_model"])
        self.ff = FeedForward(cfg)

    def forward(self, x, src_mask):
        shortcut = x
        x = self.multi_head_attention(x, src_mask)
        x = x + shortcut 
        x = self.norm1(x)

        shortcut = x
        x = self.ff(x)
        x = x + shortcut
        return self.norm2(x)
        

class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.masked_multi_head_attention = MaskedMultiHeadAttention(cfg)
        self.multi_head_cross_attention = MultiHeadCrossAttention(cfg)
        self.norm1 = LayerNorm(cfg["d_model"])
        self.norm2 = LayerNorm(cfg["d_model"])
        self.norm3 = LayerNorm(cfg["d_model"])
        self.ff = FeedForward(cfg)
        

    def forward(self, memory, x, src_mask, tgt_mask):
        shortcut = x
        x = self.masked_multi_head_attention(x, tgt_mask)
        x = x + shortcut
        x = self.norm1(x)

        shortcut = x
        x = self.multi_head_cross_attention(x, memory, src_mask)
        x = x + shortcut
        x = self.norm2(x)

        shortcut = x
        x = self.ff(x)
        x = x + shortcut

        return self.norm3(x)

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg["n_layers"])])
    
    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg["n_layers"])])

    def forward(self, memory, d_in, src_mask, tgt_mask):
        for block in self.decoder_blocks:
            d_in = block(memory, d_in, src_mask, tgt_mask)
        return d_in

def _init_positional_encoding(context_size, d_model):
    pos = torch.arange(context_size, dtype=torch.float32).unsqueeze(1)
    div = 10000**(torch.arange(0, d_model, step=2, dtype=torch.float32) / d_model)
    pe = torch.zeros((context_size, d_model))
    pe[:, 0::2] = torch.sin(pos/div)
    pe[:, 1::2] = torch.cos(pos/div)

    return pe

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
       

    def forward(self, src, dec_in, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        
        return self.decoder(memory, dec_in, src_mask, tgt_mask)




class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_token"])
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg["d_model"] ** -0.5)
        with torch.no_grad():
            self.embedding.weight[cfg["pad_token"]].zero_()
            
        self.temp = cfg["temperature"]

        self.d_model = cfg["d_model"]
        self.register_buffer("pe", _init_positional_encoding(cfg["context_size"], cfg["d_model"]))
        self.transformer_block = TransformerBlock(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.linear = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

        ## weight_tying
        self.linear.weight = self.embedding.weight


    def forward(self, src, dec_in, src_mask, tgt_mask):
        _, seq_size_src = src.shape
        _, seq_size_dec = dec_in.shape
        scale = math.sqrt(self.d_model)

        src = self.dropout(self.embedding(src) * scale + self.pe[:seq_size_src])
        dec_in = self.dropout(self.embedding(dec_in) * scale + self.pe[:seq_size_dec])

        x = self.transformer_block(src, dec_in, src_mask, tgt_mask)
        x = self.linear(x)
        return x




### V2 ####

'''
Main Changes:
    - use of F.scaled_dot_product_attention OK
    - PE => RoPE NOK
    - LayerNorm => RMSNorm OK
    - FF => FeedForward SWIGLU OK
    - Pre layer Norm OK

'''

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.g

class FeedForwardSwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.w2 = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)
        self.w3 = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, max_len, base=10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))  
        pos = torch.arange(max_len).float()                                       
        angles = pos[:, None] * inv_freq[None, :]                                
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):                      # x: (B, n_head, T, d_head)
        T = x.size(-2)

        cos, sin = self.cos[:T], self.sin[:T]

        q1 = x[..., 0::2]
        q2 = x[..., 1::2]

        out1 = q1 * cos - q2 * sin                 
        out2 = q1 * sin + q2 * cos                 

        out = torch.stack([out1, out2], dim=-1)    
        return out.flatten(-2)



class MaskedMultiHeadAttentionV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_head"] == 0, "the n_head must eb a dividor of d_model"

        self.d_in = cfg["d_model"]
        self.d_out = cfg["d_model"]
        self.n_head = cfg["n_head"]
        self.d_head = cfg["d_model"] // cfg["n_head"]

        self.W_q = nn.Linear(cfg["d_model"], cfg["d_model"], bias = cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["d_model"], cfg["d_model"], bias = cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["d_model"], cfg["d_model"], bias = cfg["qkv_bias"])
        self.W_o = nn.Linear(cfg["d_model"], cfg["d_model"])
        self.dropout = cfg["dropout"]
        self.rope = RotaryEmbedding(self.d_head, cfg["context_size"])

    def forward(self, x, tgt_mask):

        batch, seq_size, d_model = x.shape

        Q = self.W_q(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)

        Q, K = self.rope(Q), self.rope(K)

        ### We replace the whole attention score cumputation by a premade torch method

        attention_score = F.scaled_dot_product_attention(Q, K, V, attn_mask=tgt_mask, dropout_p= self.dropout if self.training else 0.0)

        attention_score = attention_score.transpose(1,2).contiguous().view(batch, seq_size, d_model)

        return self.W_o(attention_score)
        
class MultiHeadAttentionV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_head"] == 0, "the number of head must be a dividor of embedding size"

        self.d_head = cfg["d_model"] // cfg["n_head"]
        self.n_head = cfg["n_head"]
        self.d_in = cfg["d_model"]
        self.d_out = cfg["d_model"]

        self.dropout = cfg["dropout"]

        self.W_q = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_o = nn.Linear(cfg["d_model"], cfg["d_model"])
        self.rope = RotaryEmbedding(self.d_head, cfg["context_size"])

    def forward(self, x, src_mask):
        batch, seq_size, d_model = x.shape

        # dim: (b, seq_size, d_model)
       
        batch, seq_size, d_model = x.shape

        Q = self.W_q(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1, 2)
        Q, K = self.rope(Q), self.rope(K)

        ### We replace the whole attention score cumputation by a premade torch method

        attention_score = F.scaled_dot_product_attention(Q, K, V, attn_mask=src_mask, dropout_p= self.dropout if self.training else 0.0)

        attention_score = attention_score.transpose(1,2).contiguous().view(batch, seq_size, d_model)

        return self.W_o(attention_score)






class MultiHeadCrossAttentionV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["d_model"] % cfg["n_head"] == 0, "the number of head must be a dividor of embedding size"

        self.d_head = cfg["d_model"] // cfg["n_head"]
        self.n_head = cfg["n_head"]
        self.d_in = cfg["d_model"]
        self.d_out = cfg["d_model"]

        self.dropout = cfg["dropout"]

        self.W_q = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["d_model"], cfg["d_model"], bias=cfg["qkv_bias"])
        self.W_o = nn.Linear(cfg["d_model"], cfg["d_model"])


    def forward(self, x, memory, src_mask):
        batch, seq_size, d_model = x.shape
        _, seq_size_memory, _ = memory.shape

        # dim: (b, seq_size, d_model)
        Q = self.W_q(x).view(batch, seq_size, self.n_head, self.d_head).transpose(1,2)
        K = self.W_k(memory).view(batch, seq_size_memory, self.n_head, self.d_head).transpose(1,2)
        V = self.W_v(memory).view(batch, seq_size_memory, self.n_head, self.d_head).transpose(1,2)


        att_score = F.scaled_dot_product_attention(Q, K, V, attn_mask=src_mask, dropout_p=self.dropout if self.training else 0.0)
        att_score = att_score.transpose(1, 2).contiguous().view(batch, seq_size, d_model)

        return self.W_o(att_score)

class EncoderBlockV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.multi_head_attention = MultiHeadAttentionV2(cfg)
        self.norm1 = RMSNorm(cfg["d_model"])
        self.norm2 = RMSNorm(cfg["d_model"])
        self.ff = FeedForwardSwiGLU(cfg)

    def forward(self, x, src_mask):
        x = x + self.multi_head_attention(self.norm1(x), src_mask)
        x = x + self.ff(self.norm2(x))
        return x
        
        

class DecoderBlockV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.masked_multi_head_attention = MaskedMultiHeadAttentionV2(cfg)
        self.multi_head_cross_attention = MultiHeadCrossAttentionV2(cfg)
        self.norm1 = RMSNorm(cfg["d_model"])
        self.norm2 = RMSNorm(cfg["d_model"])
        self.norm3 = RMSNorm(cfg["d_model"])
        self.ff = FeedForwardSwiGLU(cfg)
        

    def forward(self, memory, x, src_mask, tgt_mask):
        '''We apply prenorm instead of Post norm
        PostNorm: x = norm(x + attention(x))
        PreNorm! x = x + attention(norm(x))

        Goal: Make training more stable when you stack a lot of transformers blocks
        '''

        x = x + self.masked_multi_head_attention(self.norm1(x), tgt_mask)
        x = x + self.multi_head_cross_attention(self.norm2(x), memory, src_mask)
        x = x + self.ff(self.norm3(x))
        return x

class EncoderV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlockV2(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["d_model"])

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        return self.final_norm(x)

class DecoderV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlockV2(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["d_model"])

    def forward(self, memory, d_in, src_mask, tgt_mask):
        for block in self.decoder_blocks:
            d_in = block(memory, d_in, src_mask, tgt_mask)
        return self.final_norm(d_in)

class TransformerBlockV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = EncoderV2(cfg)
        self.decoder = DecoderV2(cfg)
       

    def forward(self, src, dec_in, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        
        return self.decoder(memory, dec_in, src_mask, tgt_mask)


class TransformerV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_token"])
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg["d_model"] ** -0.5)
        with torch.no_grad():
            self.embedding.weight[cfg["pad_token"]].zero_()
                
        self.d_model = cfg["d_model"]
        self.transformer_block = TransformerBlockV2(cfg)
        self.linear = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
    
        ## weight_tying
        self.linear.weight = self.embedding.weight
        self.dropout = nn.Dropout(cfg["dropout"])
    
    
    def forward(self, src, dec_in, src_mask, tgt_mask):
        _, seq_size_src = src.shape
        _, seq_size_dec = dec_in.shape
        scale = math.sqrt(self.d_model)

        src = self.dropout(self.embedding(src) * scale)
        dec_in = self.dropout(self.embedding(dec_in) * scale)

        x = self.transformer_block(src, dec_in, src_mask, tgt_mask)
        x = self.linear(x)
        return x
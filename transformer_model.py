import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainingConfig
from typing import Optional, List
from torch.types import Tuple

class RotaryEmbedding(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        assert config.model_dim % config.n_query_heads == 0
        head_dim = config.model_dim // config.n_query_heads

        numerator = torch.arange(0, head_dim, 2, dtype=torch.float32, device=config.device)
        freqs = 1.0 / (config.rope_theta ** (numerator / head_dim))
        t = torch.arange(config.context_length, device=config.device)

        freqs = torch.outer(t, freqs)
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    def forward(self, query, key, start_pos=None):
        q_shape = query.shape
        k_shape = key.shape
        assert q_shape[1] == k_shape[1]
        sequence_length = q_shape[1]

        query = torch.view_as_complex(query.float().reshape(*q_shape[:-1], -1, 2))
        key = torch.view_as_complex(key.float().reshape(*k_shape[:-1], -1, 2))

        freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(2)

        if sequence_length < self.config.context_length:
            assert start_pos is not None
            freqs_complex = freqs_complex[:, start_pos:start_pos+sequence_length, :, :]

        query = torch.view_as_real(query * freqs_complex).reshape(*q_shape)
        key = torch.view_as_real(key * freqs_complex).reshape(*k_shape)

        return query, key


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)

class KeyValueCache:
    def __init__(self, config: TrainingConfig):
        assert config.model_dim % config.n_key_value_heads == 0
        head_dim = config.model_dim // config.n_query_heads

        self.keys = torch.zeros((config.batch_size, config.context_length, config.n_key_value_heads, head_dim), device=config.device)
        self.values = torch.zeros((config.batch_size, config.context_length, config.n_key_value_heads, head_dim), device=config.device)

    def update(self, batch_size, start_pos, key, value):
        assert key.shape[1] == value.shape[1]
        sequence_length = key.shape[1]
        self.keys[:batch_size, start_pos: start_pos+sequence_length] = key
        self.values[:batch_size, start_pos: start_pos+sequence_length] = value

    def get(self, batch_size, start_pos, sequence_length):
        keys = self.keys[:batch_size, :start_pos + sequence_length]
        values = self.values[:batch_size, :start_pos + sequence_length]
        return keys, values

class DumbleLLM(nn.Module):
    def __init__(self, config: TrainingConfig, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        #self.pos_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        # TODO: weight initialization
        # TODO: weight tying

    def forward(self, tokens, targets=None, start_pos=None):
        x = self.token_embedding(tokens)
        x, _ = self.blocks((x, start_pos))
        logits = self.output(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    # TODO: top-p sampling
    @torch.inference_mode()
    def top_p_sampling(self, prompts: List[str], temperature: float = 0.6, p: float = 0.9, max_length: Optional[int] = None):

        if max_length is None:
            max_length = self.config.context_length

        for p in prompts:
            tokens = torch.tensor(self.tokenizer.encode(p, add_bos=True, add_eps=False)).view(1, -1)
            tokens = tokens.to(self.config.device)
        

    @torch.inference_mode()
    def generate(self, tokens, max_length):
        res = tokens
        assert res.shape[1] <= self.config.context_length
        
        pos = 0
        while res.shape[1] < max_length:
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                logits, _ = self.forward(res, start_pos=pos)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            idx = torch.multinomial(topk_probs, 1) # (B, 1)
            xcol = torch.gather(topk_indices, -1, idx)
            res = torch.cat((res, xcol), dim=1)
            pos = res.shape[1]

        return res

class CausalSelfAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        assert self.config.model_dim % self.config.n_query_heads == 0
        assert self.config.model_dim % self.config.n_key_value_heads == 0

        self.head_dim = self.config.model_dim // self.config.n_query_heads

        #self.att_w = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)

        self.wq = nn.Linear(config.model_dim, config.n_query_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.model_dim, config.n_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.model_dim, config.n_key_value_heads * self.head_dim, bias=False)

        self.pos_embedding = RotaryEmbedding(config)

        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.n_query_heads * self.head_dim, config.model_dim, bias=False)

        if self.config.use_kv_cache:
            self.kv_cache = KeyValueCache(self.config)

    def forward(self, x, start_pos=None):
        batch_size, sequence_length, model_dim = x.shape

        #query, key, value = self.att_w(x).split(self.config.model_dim, dim=2)

        query, key, value = self.wq(x), self.wk(x), self.wv(x)

        query = query.view(batch_size, sequence_length, self.config.n_query_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.config.n_key_value_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.config.n_key_value_heads, self.head_dim)

        query, key = self.pos_embedding(query, key, start_pos)

        # use key value cache at inference
        if self.config.use_kv_cache and not self.training and start_pos is not None:
            self.kv_cache.update(batch_size, start_pos, key, value)
            key, value = self.kv_cache.get(batch_size, start_pos, sequence_length)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        enable_gqa = False
        if self.config.n_query_heads != self.config.n_key_value_heads:
            assert self.config.n_query_heads % self.config.n_key_value_heads == 0
            enable_gqa = True

        # Flash Attention
        x = F.scaled_dot_product_attention(query, key, value, dropout_p=self.config.dropout if self.training else 0, is_causal=True, enable_gqa=enable_gqa)
        x = x.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        x = self.dropout(self.output(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.att_norm = nn.RMSNorm(config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.RMSNorm(config.model_dim)
        self.feed_forward = FeedForward(config)
    
    def forward(self, args):
        x, start_pos = args
        x = x + self.attention(self.att_norm(x), start_pos=start_pos)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, start_pos


class FeedForward(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.w1 = nn.Linear(config.model_dim, 4 * config.model_dim, bias=False)
        self.swiglu = SwiGLU()
        self.w2 = nn.Linear(config.model_dim, 2 * config.model_dim, bias=False)
        self.w3 = nn.Linear(2 * config.model_dim, config.model_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = self.swiglu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        x = self.dropout(x)
        return x

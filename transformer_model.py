import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainingConfig
from typing import Optional, List
from torch.types import Tuple
import math

from torch.nn.init import _calculate_fan_in_and_fan_out

class RotaryEmbedding(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        assert config.model_dim % config.n_query_heads == 0
        head_dim = config.model_dim // config.n_query_heads

        numerator = torch.arange(0, head_dim, 2, dtype=torch.float32, device=config.device)
        freqs = 1.0 / (config.rope_theta ** (numerator / head_dim))
        t = torch.arange(2* config.context_length, device=config.device)

        freqs = torch.outer(t, freqs)
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    def forward(self, query, key, start_pos=None):
        q_shape = query.shape
        k_shape = key.shape

        query = torch.view_as_complex(query.float().reshape(*q_shape[:-1], -1, 2))
        key = torch.view_as_complex(key.float().reshape(*k_shape[:-1], -1, 2))

        freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(2)

        if start_pos is not None: # inference
            freqs_complex_query = freqs_complex[:, start_pos:start_pos + q_shape[1], :, :]
            freqs_complex_key = freqs_complex[:, start_pos:start_pos + k_shape[1], :, :]
        else:
            freqs_complex_query = freqs_complex[:, :q_shape[1], :, :]
            freqs_complex_key = freqs_complex[:, :k_shape[1], :, :]

        query = torch.view_as_real(query * freqs_complex_query).reshape(*q_shape)
        key = torch.view_as_real(key * freqs_complex_key).reshape(*k_shape)

        return query, key


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)

class KeyValueCache:
    def __init__(self, config: TrainingConfig):
        self.initialized = False
        assert config.model_dim % config.n_key_value_heads == 0
        head_dim = config.model_dim // config.n_query_heads

        self.keys = torch.zeros((config.batch_size, config.context_length, config.n_key_value_heads, head_dim), device=config.device)
        self.values = torch.zeros((config.batch_size, config.context_length, config.n_key_value_heads, head_dim), device=config.device)


    def update(self, batch_size, start_pos, key, value):
        assert key.shape[1] == value.shape[1]
        sequence_length = key.shape[1]
        self.initialized = True

        self.keys[:batch_size, start_pos: start_pos + sequence_length] = key
        self.values[:batch_size, start_pos: start_pos + sequence_length] = value

    def get(self, batch_size, start_pos, sequence_length):
        assert self.initialized
        keys = self.keys[:batch_size, :start_pos + sequence_length]
        values = self.values[:batch_size, :start_pos + sequence_length]
        return keys, values

class DumbleLLM(nn.Module):
    def __init__(self, config: TrainingConfig, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        # weight tying
        self.token_embedding.weight = self.output.weight

        self.apply(self._init_weights)

    def forward(self, tokens, targets=None, start_pos=None):
        x = self.token_embedding(tokens)
        x, _ = self.blocks((x, start_pos))
        logits = self.output(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss
        

    @torch.inference_mode()
    def generate(self, prompts, max_length, strategy="top_p", top_p=0.9, top_p_temperature=1):
        result_list = []

        for prompt in prompts:
            tokens = torch.tensor(self.tokenizer.encode(prompt)).view(1, -1)
            tokens = tokens.to(self.config.device)
            
            result = tokens
            assert result.shape[1] < self.config.context_length
            max_length = min(max_length, self.config.context_length)

            pos = 0
            while result.shape[1] < max_length:
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, _ = self.forward(result, start_pos=pos)
                logits = logits[:, -1, :]
                logits /= top_p_temperature
                probs = F.softmax(logits, dim=-1)

                if strategy == "top_p":
                    new_token = self._top_p_sampling(probs, top_p)
                elif strategy == "top_k":
                    new_token = self._top_k_sampling(probs)
                
                result = torch.cat((result, new_token), dim=1)
                if new_token == self.tokenizer.eos_id:
                    break
                pos = result.shape[1] - 1

            result_list.append(result)
        
        result_list = [self.tokenizer.decode(result.tolist()[0]) for result in result_list]
        result_list = [result.replace('\n', ' ') for result in result_list]
        return result_list
    
    def _top_k_sampling(self, probs):
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        idx = torch.multinomial(topk_probs, 1)
        col = torch.gather(topk_indices, -1, idx)
        return col

    def _top_p_sampling(self, probs, p=0.9):
        probs_sorted, probs_idx = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(probs, dim=-1)
        mask = cum_probs - probs_sorted > p
        probs_sorted[mask] = 0.0
        probs_sorted.div_(probs_sorted.sum(dim=-1, keepdim=True))
        idx = torch.multinomial(probs_sorted, 1)
        col = torch.gather(probs_idx, -1, idx)
        return col

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #if hasattr(module, 'SCALE_INIT'):
            #    fan_in, fan_out = _calculate_fan_in_and_fan_out(module.weight)
            #    std = math.sqrt(2.0 / float(fan_in + fan_out))
            #    std *= (2 * self.config.n_layers) ** -0.5
            #    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            #else:
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)

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
        #self.output.SCALE_INIT = 1

        if self.config.use_kv_cache:
            self.kv_cache = KeyValueCache(self.config)

    def forward(self, x, start_pos=None):
        batch_size, sequence_length, _ = x.shape

        query_length = sequence_length
        kv_length = sequence_length

        is_inference = (start_pos is not None and not self.training)
        is_initial_prompt = (is_inference and start_pos == 0)
        use_kv_cache = self.config.use_kv_cache and is_inference

        if is_inference and not is_initial_prompt:
            query_length = 1
            query = self.wq(x[:, -1, :])                                #   inference, not initial prompt
        
            if use_kv_cache:
                kv_length = 1
                key, value = self.wk(x[:, -1, :]), self.wv(x[:, -1, :]) #   inference, not initial prompt, kv cache
            else:
                key, value = self.wk(x), self.wv(x)                     #   inference, not initial prompt, no kv cache
        else:
            query, key, value = self.wq(x), self.wk(x), self.wv(x)      #   training and inference with initial prompt
        
        query = query.view(batch_size, query_length, self.config.n_query_heads, self.head_dim)
        key = key.view(batch_size, kv_length, self.config.n_key_value_heads, self.head_dim)
        value = value.view(batch_size, kv_length, self.config.n_key_value_heads, self.head_dim)


        if use_kv_cache:
            self.kv_cache.update(batch_size, start_pos, key, value)
            key, value = self.kv_cache.get(batch_size, start_pos, kv_length)

        query, key = self.pos_embedding(query, key, start_pos)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        enable_gqa = False
        if self.config.n_query_heads != self.config.n_key_value_heads:
            assert self.config.n_query_heads % self.config.n_key_value_heads == 0
            enable_gqa = True

        # Flash Attention
        x = F.scaled_dot_product_attention(query, key, value, dropout_p=self.config.dropout if self.training else 0, is_causal=True, enable_gqa=enable_gqa)
        x = x.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
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
        #self.w3.SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = self.swiglu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        x = self.dropout(x)
        return x

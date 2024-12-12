import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainingConfig


class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)

class Residual(nn.Module):
    pass


class DumbleLLM(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_embedding = nn.Embedding(config.vocab_size, config.model_dim) # TODO: replace with RotaryEmbedding
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        # TODO: weight initialization
        # TODO: weight tying

    def forward(self, tokens, targets=None):
        batch_size, sequence_length = tokens.shape
        pos = torch.arange(0, sequence_length, dtype=torch.long, device=self.config.device)

        x = self.pos_embedding(pos) + self.token_embedding(tokens)
        x = self.blocks(x)
        logits = self.output(self.norm(x))

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    @torch.inference_mode()
    def generate(self, tokens):
        pass


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.att_w = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)


    def forward(self, x):
        batch_size, sequence_length, model_dim = x.shape
        query, key, value = self.att_w(x).split(self.config.model_dim, dim=2)

        assert self.config.model_dim % self.config.n_attention_heads == 0
        assert self.config.model_dim % self.config.n_key_value_heads == 0

        att_head_dim = self.config.model_dim // self.config.n_attention_heads
        kv_head_dim = self.config.model_dim // self.config.n_key_value_heads

        query = query.view(batch_size, sequence_length, self.config.n_attention_heads, att_head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.config.n_key_value_heads, kv_head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.config.n_key_value_heads, kv_head_dim).transpose(1, 2)

        # Flash Attention
        x = F.scaled_dot_product_attention(query, key, value, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dim)
        x = self.dropout(self.output(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.att_norm = nn.RMSNorm(config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.RMSNorm(config.model_dim)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attention(self.att_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


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

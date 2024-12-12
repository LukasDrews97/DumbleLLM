from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_dim: int = 512
    #ffn_dim: int = 4096
    context_length = 512
    vocab_size: int = 512
    n_layers: int = 4
    n_attention_heads: int = 4
    n_key_value_heads: int = 4# 8
    dropout : float = 0.2
    device: str = 'cuda'
    batch_size: int = 32
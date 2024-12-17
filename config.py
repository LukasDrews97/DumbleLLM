from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # general
    device: str = 'cuda'

    # architecture
    model_dim: int = 512
    #ffn_dim: int = 4096
    context_length = 512
    vocab_size: int = 512
    n_layers: int = 4
    n_query_heads: int = 8 # 8
    n_key_value_heads: int = 4
    rope_theta: float = 50000

    # training
    batch_size: int = 32
    micro_batch_size: int = 4 # set to batch_size to disable gradient accumulation
    n_epochs: int = 1
    dropout : float = 0.1

    # inference
    use_kv_cache: bool = False

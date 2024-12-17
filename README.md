# DumbleLLM - Custom Large Language Model

This repository contains the code for a decoder-only transformer, similar to Llama, or GPT. It was trained on an English corpus built from the seven Harry Potter books.

# Technical Features

- Tokenization: Byte pair encoding (sentencepiece)
- FlashAttention, Grouped Query Attention
- Rotary Position Embeddings
- Key Value Cache
- Sampling: top-p, top-k


# Training Configuration
| **Parameter**          | **Value**   |
|-------------------------|-------------|
| Layer                  | 4           |
| Model Dimension        | 768         |
| Context Length         | 1024        |
| Attention Heads        | 8           |
| Key/Value Heads        | 4           |
| Vocabulary Size        | 32000       |
| RoPE Theta             | 10000       |

"markdown-checkbox.strikeThroughWhenChecked": true

# Roadmap
- [x] ~~Grouped Query Attention~~
- [x] ~~Rotary Position Embeddings~~
- [x] ~~Key Value Cache~~
- [ ] Enable distributed training
- [ ] Add Mixture of Experts model

# Example Prompts
TODO
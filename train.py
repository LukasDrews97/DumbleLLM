from tokenizer import Tokenizer
from transformer_model import DumbleLLM
from config import TrainingConfig
import torch

TRAINING_DATA = 'data/input.txt'




if __name__ == '__main__':
    config = TrainingConfig()

    tokenizer = Tokenizer(input_file=TRAINING_DATA, vocab_size=config.vocab_size, retrain=True)

    ids = tokenizer.encode('Hello World!')
    print(ids)

    text = tokenizer.decode(ids)
    print(text)


    model = DumbleLLM(config)
    model.to(config.device)

    print(model.forward(torch.tensor(ids, device=config.device).view(1, -1)))

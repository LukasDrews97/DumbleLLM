from tokenizer import Tokenizer
from transformer_model import DumbleLLM
from config import TrainingConfig
from data_loader import Dataset
import torch
from tqdm import tqdm

TRAINING_DATA = 'data/input.txt'


if __name__ == '__main__':
    config = TrainingConfig()

    tokenizer = Tokenizer(input_file=TRAINING_DATA, vocab_size=config.vocab_size, retrain=False)

    ids = tokenizer.encode('Hello World!')
    print(ids)

    text = tokenizer.decode(ids)
    print(text)

    dataset = Dataset(TRAINING_DATA, tokenizer, config.batch_size, config.context_length)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True
    )

    #torch.autograd.set_detect_anomaly(True)

    model = DumbleLLM(config)
    model.to(config.device)

    # TODO: add LRScheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # TODO: gradient accumulation
    for idx, (token, targets) in enumerate(train_dataloader):
        token, targets = token.to(config.device), targets.to(config.device)
        model.train()

        optimizer.zero_grad()
        logits, loss = model(token, targets)
        loss.backward()
        optimizer.step()
        print(f"step {idx}, loss: {loss.item()}")
        
        with torch.no_grad():
            model.eval()
            pass

    #print(model.forward(torch.tensor(ids, device=config.device).view(1, -1)))

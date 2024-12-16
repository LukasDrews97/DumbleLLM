from tokenizer import Tokenizer
from transformer_model import DumbleLLM
from config import TrainingConfig
from data_loader import Dataset
import torch
from tqdm import tqdm

TRAINING_DATA = 'data/input.txt'



if __name__ == '__main__':
    config = TrainingConfig()

    tokenizer = Tokenizer(input_file=TRAINING_DATA, vocab_size=config.vocab_size, retrain=True)

    #ids = tokenizer.encode('Hello World!')
    #print(ids)

    #text = tokenizer.decode(ids)
    #print(text)

    dataset = Dataset(TRAINING_DATA, tokenizer, config.batch_size, config.context_length)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.micro_batch_size,
            shuffle=True,
            drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config.micro_batch_size,
            shuffle=False,
            drop_last=True
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    model = DumbleLLM(config, tokenizer)
    model.to(config.device)
    model = torch.compile(model)

    print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    #import sys; sys.exit()

    # TODO: add LRScheduler, weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    n_grad_accum_steps = config.batch_size // config.micro_batch_size

    for epoch in range(config.n_epochs):
        train_loss_sum = 0
        for idx, (token, targets) in enumerate(train_dataloader):
            token, targets = token.to(config.device), targets.to(config.device)
            model.train()
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                logits, loss = model(token, targets)

            train_loss_sum += loss.detach()
            loss.backward()

            # accumulate gradients
            if (idx+1) % n_grad_accum_steps == 0 or (idx+1 == len(train_dataloader)): # optimize after one full batch was processed
                optimizer.step()
                optimizer.zero_grad()
                print(f"step {idx+1}, training loss after {n_grad_accum_steps} grad accum. steps: {loss.item():.2f}")
            
            
            if idx % 100 == 0:
                with torch.inference_mode():
                    model.eval()
                    test_loss_sum = 0
                    for test_token, test_targets in test_dataloader:
                        test_token, test_targets = test_token.to(config.device), test_targets.to(config.device)
                        with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                            _, test_loss = model(test_token, test_targets)
                            test_loss_sum += test_loss
                    print(f"epoch: {epoch} idx: {idx} TEST LOSS: {(test_loss_sum / len(test_dataloader)):.2f}")
            
            
        
        print(f"EPOCH {epoch} AVERAGE TRAIN LOSS: {(train_loss_sum / len(train_dataloader)):.2f}")


        # TODO: fix newline token
        print("STARTING TEXT GENERATION")
        model.eval()
        with torch.inference_mode():
            start = "Harry saw that "
            tokens = torch.tensor(tokenizer.encode(start)).view(1, -1)
            #print(tokens.shape)
            tokens = tokens.to(config.device)
            logits = model.generate(tokens, max_length=100)
            res = tokenizer.decode(logits.tolist())[0]
            res = res.replace('\n', ' ')
            print(res)

    # TODO: add checkpoints for model training
    # TODO: load model from weights
    # TODO: HellaSwag, Perplexity

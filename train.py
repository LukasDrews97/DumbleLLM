from tokenizer import Tokenizer
from transformer_model import DumbleLLM
from config import TrainingConfig
from data_loader import Dataset
from logger import Logger
import torch
from tqdm import tqdm
import os

TRAINING_DATA = 'data/input.txt'
MODEL_WEIGHTS_DIR = "saved"
LOGGING_DIR = "saved" 



if __name__ == '__main__':

    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)

    config = TrainingConfig()
    tokenizer = Tokenizer(input_file=TRAINING_DATA, vocab_size=config.vocab_size, retrain=True)

    logger = Logger(path=LOGGING_DIR)

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

    #checkpoint = torch.load(f"{MODEL_WEIGHTS_DIR}/state.pt", weights_only=True)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.compile(model)

    print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # TODO: add LRScheduler, weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                print(f"step {idx+1}, training loss after {n_grad_accum_steps} grad accum. steps: {loss.item():.2f} norm: {norm:.4f}")
                logger.log_train_loss(epoch+1, idx, loss.item(), config.batch_size)
            
        
        # calculate test loss after each epoch
        with torch.inference_mode():
            model.eval()
            test_loss_sum = 0
            for test_token, test_targets in test_dataloader:
                test_token, test_targets = test_token.to(config.device), test_targets.to(config.device)
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    _, test_loss = model(test_token, test_targets)
                    test_loss_sum += test_loss

            avg_test_loss = (test_loss_sum / len(test_dataloader)).item()
            print(f"epoch: {epoch+1}/{config.n_epochs} AVG TEST LOSS: {avg_test_loss:.2f}")
            
            
        avg_train_loss = (train_loss_sum / len(train_dataloader)).item()
        print(f"EPOCH {epoch+1}/{config.n_epochs} AVERAGE TRAIN LOSS: {avg_train_loss:.2f}")

        logger.log_loss_epoch(epoch+1, avg_train_loss, avg_test_loss)

        # save current model
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': (train_loss_sum / len(train_dataloader)),
            'test_loss': (test_loss_sum / len(test_dataloader))
        }, f"{MODEL_WEIGHTS_DIR}/state.pt")

        # generate text
        print("STARTING TEXT GENERATION")
        model.eval()
        with torch.inference_mode():
            prompts = ["Harry saw that ", "Dumbledore "]
            results = model.generate(prompts, max_length=config.context_length, strategy="top_p")
            for idx, res in enumerate(results):
                print(f"PROMPT {idx+1}:")
                print(res)
                print("=============================================")
            
            for prompt, result in zip(prompts, results):
                logger.log_prompt(epoch+1, prompt, result)
        
        logger.save_to_file()


    # TODO: HellaSwag, Perplexity
    # TODO: README
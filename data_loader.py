import torch.utils.data as data
import torch

class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer, batch_size, context_length):
        super().__init__()
        self.context_length = context_length

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokens = tokenizer.encode(text)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (batch_size * context_length)} batches')

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokens[idx*self.context_length : (1+idx)*self.context_length])
        targets = torch.tensor(self.tokens[idx*self.context_length+1 : (1+idx)*self.context_length+1])
        return tokens, targets
    
    def __len__(self):
        return len(self.tokens) // self.context_length

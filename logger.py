import os
from datetime import datetime
import pandas as pd

class Logger:
    def __init__(self, path=None):
        if path is None:
            path = os.path.abspath(os.getcwd())
        self.path = path
        os.makedirs(path, exist_ok=True)

        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.batch_count = 0

        self.train_loss = pd.DataFrame({"epoch": pd.Series([], dtype="int32"), "step": pd.Series([], dtype="int32"), "used_batches": pd.Series([], dtype="int32"), "loss": pd.Series([], dtype="float32")})
        self.epoch_loss = pd.DataFrame({"epoch": pd.Series([], dtype="int32"), "avg_train_loss": pd.Series([], dtype="float32"), "avg_test_loss": pd.Series([], dtype="float32")})
        self.prompts = pd.DataFrame({"epoch": pd.Series([], dtype="int32"), "prompt": [], "output": []})


    def log_train_loss(self, epoch, step, loss, batch_size):
        self.batch_count += batch_size
        row = pd.DataFrame([{"epoch": int(epoch), "step": step, "used_batches": self.batch_count, "loss": loss}])
        self.train_loss = pd.concat([self.train_loss, row], ignore_index=True)

    def log_loss_epoch(self, epoch, train_loss, test_loss):
        row = pd.DataFrame([{"epoch": int(epoch), "avg_train_loss": train_loss, "avg_test_loss": test_loss}])
        self.epoch_loss = pd.concat([self.epoch_loss, row], ignore_index=True)
    
    def log_prompt(self, epoch, prompt, output):
        row = pd.DataFrame([{"epoch": int(epoch), "prompt": prompt, "output": output}])
        self.prompts = pd.concat([self.prompts, row], ignore_index=True)

    def save_to_file(self):
        self.train_loss.to_csv(f"{self.path}/train_loss_{self.current_time}.csv", index=False)
        self.epoch_loss.to_csv(f"{self.path}/epoch_loss_{self.current_time}.csv", index=False)
        self.prompts.to_csv(f"{self.path}/prompts_{self.current_time}.csv", index=False)

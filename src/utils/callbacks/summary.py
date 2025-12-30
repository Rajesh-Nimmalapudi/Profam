import csv
import os
import torch
from lightning.pytorch.callbacks import Callback

class SummaryCSVCallback(Callback):
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.headers = ["Epoch", "Avg Loss", "Avg Accuracy", "End Step"]
        
        # Initialize file if it doesn't exist
        if not os.path.exists(self.output_path):
            with open(self.output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve metrics
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        step = trainer.global_step
        
        # Loss and accuracy are logged as 'train/loss' and 'train/aa_accuracy'
        # Note: Depending on how they are averaged, we might want the epoch-level mean
        # Lightning's callback_metrics usually contains the latest logged value.
        # However, we want the mean for the epoch.
        # Fortunately, BaseFamilyLitModule logs with on_step=True, on_epoch=False for training.
        # We might need to handle the averaging ourselves or ensure Lightning logs epoch-level means.
        
        loss = metrics.get("train/loss_epoch") or metrics.get("train/loss", 0.0)
        acc = metrics.get("train/aa_accuracy_epoch") or metrics.get("train/aa_accuracy", 0.0)
        
        if torch.is_tensor(loss):
            loss = loss.item()
        if torch.is_tensor(acc):
            acc = acc.item()

        # Append to CSV
        with open(self.output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{loss:.4f}", f"{acc:.4f}", step])

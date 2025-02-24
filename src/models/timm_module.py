import pytorch_lightning as pl
import torch
import timm
import torch.nn.functional as F
from torchmetrics import Accuracy
import json
import os

class TIMMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        pretrained: bool = True,
        data_dir: str = None  # Add data_dir parameter
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Increase model capacity for combined dataset
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3  # Add dropout
        )
        
        # Load class counts and compute weights if data_dir is provided
        if data_dir and os.path.exists(os.path.join(data_dir, 'class_counts.json')):
            with open(os.path.join(data_dir, 'class_counts.json'), 'r') as f:
                class_counts = json.load(f)
            
            # Calculate class weights
            total_samples = sum(class_counts.values())
            class_weights = [total_samples / (len(class_counts) * count) 
                           for count in class_counts.values()]
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Use weighted loss if weights are available
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        else:
            loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer 

    def on_save_checkpoint(self, checkpoint):
        """Save additional information to checkpoint."""
        checkpoint["model_name"] = self.hparams.model_name
        checkpoint["num_classes"] = self.hparams.num_classes 
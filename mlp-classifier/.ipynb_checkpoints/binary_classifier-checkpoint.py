import torch
from torch import nn
from torch.functional import Tensor
from sklearn.metrics import classification_report
from pytorch_lightning import LightningModule


class LinearClassifier(LightningModule):
    def __init__(
        self,
        input_dim: int = 172,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        # Binary Classifier Layers
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Binary criterion
        self.criterion = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=1e-3, weight_decay=1e-3)

    def training_step(self, batch: list, batch_idx: int):
        embs, labels = batch
        preds = self.decoder(embs)
        loss = self.criterion(preds, labels.float())
        self.log("Loss/train", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: list, batch_idx: int):
        embs, labels = batch

        with torch.no_grad():
            preds = self.decoder(embs)
            loss = self.criterion(preds, labels.float())

        self.log("Loss/validation", loss)
        return labels, torch.round(preds.sigmoid())

    def validation_epoch_end(self, batch_parts):
        labels = torch.cat([label for label, _ in batch_parts]).cpu()
        preds = torch.cat([pred for _, pred in batch_parts]).cpu()
        report = classification_report(
            labels, preds
        )
        print(report)

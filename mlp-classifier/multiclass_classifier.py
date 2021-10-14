import torch
from torch import nn
from torch.functional import Tensor
from sklearn.metrics import classification_report
from pytorch_lightning import LightningModule


class MultiClassClassifier(LightningModule):
    def __init__(
        self,
        input_dim: int = 172,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        # Multiclass Classifier
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # Multiclass Criterion
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=1e-3, weight_decay=1e-3)

    def training_step(self, batch: list, batch_idx: int):
        embs, labels = batch
        preds = self.decoder(embs)
        loss = self.criterion(preds, labels.long().squeeze())
        self.log("Loss/train", loss, on_epoch=True)

        return loss

    def validation_step(self, batch: list, batch_idx: int):
        embs, labels = batch

        with torch.no_grad():
            preds = self.decoder(embs)
            loss = self.criterion(preds, labels.long().squeeze())

        self.log("Loss/validation", loss)
        return labels, self.softmax(preds)

    def validation_epoch_end(self, batch_parts):
        labels = torch.cat([label for label, _ in batch_parts]).cpu()
        preds = torch.cat([torch.argmax(pred, dim=1) for _, pred in batch_parts]).cpu()
        report = classification_report(labels, preds)
        print(report)

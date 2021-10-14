import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class DataModule(LightningDataModule):
    def __init__(
        self,
        DATA_PATH=None,
        LABEL_PATH=None,
        batch_size=30,
        train_size=0.6,
        num_workers=1,
        multiclass=False,
    ):
        super().__init__()
        self.DATA_PATH = DATA_PATH
        self.LABEL_PATH = LABEL_PATH
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self.multiclass = multiclass

    def prepare_data(self):
        self.df = pd.read_csv(self.DATA_PATH, header=None).to_numpy()
        self.labels = pd.read_csv(self.LABEL_PATH, header=None).to_numpy()
        if not self.multiclass:
            self.labels = (self.labels != 0)*1

    def setup(self, stage=None):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df, self.labels, train_size=self.train_size, shuffle=True
        )
        self.train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        self.val = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

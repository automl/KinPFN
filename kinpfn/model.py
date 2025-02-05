import warnings

import torch

import kinpfn
from kinpfn import utils

from .train_kinpfn import train_kinpfn


class KINPFN(torch.nn.Module):

    def __init__(
        self,
        dataset_name=None,
        seq_len=None,
        epochs=None,
        buckets=None,
        steps=None,
        batch_size=None,
        lr=None,
        emsize=None,
        nhead=None,
        nhid=None,
        nlayers=None,
        dropout=None,
        weight_decay=None,
        model_path=None,
    ):
        super(KINPFN, self).__init__()

        self.model = None

        if model_path is not None:
            try:
                self.model = torch.load(f"{model_path}")
                self.model.eval()
                print(f"Model loaded from path {model_path}.")
            except FileNotFoundError:
                warnings.warn(f"Model path {model_path} not found.")
        else:
            if any(
                param is None
                for param in [
                    dataset_name,
                    seq_len,
                    epochs,
                    buckets,
                    steps,
                    batch_size,
                    lr,
                    emsize,
                    nhead,
                    nhid,
                    nlayers,
                    dropout,
                    weight_decay,
                ]
            ):
                raise ValueError(
                    "All training parameters must be provided if model_file_name is not specified."
                )

            self.dataset_name = dataset_name
            self.seq_len = seq_len
            self.epochs = epochs
            self.buckets = buckets
            self.steps = steps
            self.batch_size = batch_size
            self.lr = lr
            self.emsize = emsize
            self.nhead = nhead
            self.nhid = nhid
            self.nlayers = nlayers
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.model = None

    def train_new_model(self, get_batch_function, is_neps_run=False):
        if self.model is not None:
            warnings.warn(
                "Model is already loaded. To retrain, create a new instance without specifying model_file_name."
            )
            return

        train_results = train_kinpfn(
            get_batch_function=get_batch_function,
            seq_len=self.seq_len,
            num_features=1,
            hps=None,
            epochs=self.epochs,
            buckets=self.buckets,
            steps=self.steps,
            batch_size=self.batch_size,
            lr=self.lr,
            emsize=self.emsize,
            nhead=self.nhead,
            nhid=self.nhid,
            nlayers=self.nlayers,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            is_neps_run=is_neps_run,
        )
        self.model = train_results[2]
        self.model.eval()
        print("Model trained and saved successfully.")
        return train_results

import neps
import random
import logging
import time
import os
import torch
import numpy as np
import polars as pl
from neps.plot.tensorboard_eval import tblogger

from kinpfn.priors import Batch
from kinpfn.model import KINPFN
from kinpfn.prior_kinpfn import get_batch_multi_modal_distribution_prior

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_dataset_size(val_set_dir):
    dataset_size = 0
    for subdir, _, files in os.walk(val_set_dir):
        for file in files:
            if file.endswith(".csv"):
                dataset_size += 1
    return dataset_size


def get_batch_testing_folding_times(val_set_dir, seq_len=100, num_features=1, **kwargs):
    dataset_size = get_dataset_size(val_set_dir)

    x = torch.zeros(seq_len, dataset_size, num_features)
    y = torch.zeros(seq_len, dataset_size)

    batch_index = 0
    for subdir, _, files in os.walk(val_set_dir):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(subdir, file)
                data = pl.read_csv(
                    path,
                    has_header=False,
                    columns=[2],
                    n_rows=seq_len,
                )

                folding_times = data["column_3"].to_numpy()

                sorted_folding_times = np.sort(folding_times)

                # Filter out points where x > 10^15 and x < 10^-6
                valid_indices = (sorted_folding_times <= 10**15) & (
                    sorted_folding_times >= 10**-6
                )
                sorted_folding_times = sorted_folding_times[valid_indices]

                # Adjust the sequence length by sampling if bounds were exceeded
                current_seq_len = len(sorted_folding_times)
                if current_seq_len <= 0:
                    continue

                if current_seq_len < seq_len:
                    # Repeat the sorted_folding_times and cdf to match the sequence length (Oversampling) (only if bounds were exceeded)
                    repeat_factor = seq_len // current_seq_len + 1
                    sorted_folding_times = np.tile(sorted_folding_times, repeat_factor)[
                        :seq_len
                    ]
                else:
                    sorted_folding_times = sorted_folding_times[:seq_len]

                x[:, batch_index, 0] = torch.tensor(np.zeros(seq_len))
                y[:, batch_index] = torch.tensor(sorted_folding_times)
                batch_index += 1

    y = torch.log10(y)

    return Batch(x=x, y=y, target_y=y)


def neps_model_validation_pipeline(trained_model):

    seq_len = 1000

    # Hardcode train indices for comparability
    true_indices = [17, 89, 207, 315, 456, 578, 634, 762, 874, 951]
    train_indices = torch.zeros(seq_len, dtype=torch.bool)
    train_indices[true_indices] = True

    val_set_dir = "../neps_validation_set"

    batch = get_batch_testing_folding_times(val_set_dir=val_set_dir, seq_len=seq_len)
    dataset_size = get_dataset_size(val_set_dir)

    x = batch.x
    y_folding_times = batch.y
    target_y_folding_times = batch.target_y

    mae_losses = []
    mean_nll_losses = []

    for i in range(dataset_size):
        batch_index = i

        # Create the training and test data
        train_x = x[train_indices, batch_index]
        train_y_folding_times = y_folding_times[train_indices, batch_index]

        test_x = x[:, batch_index]
        test_y_folding_times = target_y_folding_times[:, batch_index]

        with torch.no_grad():
            # we add our batch dimension, as our transformer always expects that
            logits = trained_model(
                train_x[:, None], train_y_folding_times[:, None], test_x[:, None]
            )

        ### CALCULATE GROUND TRUTH CDF WHICH WE WANT TO PREDICT
        test_y_folding_times_sorted, _ = torch.sort(test_y_folding_times)
        ground_truth_sorted_folding_times = test_y_folding_times.argsort()
        ground_truth_cdf = torch.arange(
            1, len(ground_truth_sorted_folding_times) + 1
        ) / len(ground_truth_sorted_folding_times)

        ### PFN PREDICTION
        predicted_cdf = (trained_model.criterion.cdf(logits, test_y_folding_times))[0][
            0
        ]

        test_x = test_x.cpu().numpy()
        test_y_folding_times = test_y_folding_times.cpu().numpy()
        train_x = train_x.cpu().numpy()
        train_y_folding_times = train_y_folding_times.cpu().numpy()

        single_absolute_error = np.abs(predicted_cdf - ground_truth_cdf)
        mae = single_absolute_error.mean()
        mae_losses.append(mae)

        nll_loss = trained_model.criterion.forward(
            logits=logits, y=test_y_folding_times_sorted
        )
        mean_nll_loss = nll_loss.mean()
        mean_nll_losses.append(mean_nll_loss)

    mae_losses_stacked = torch.stack(mae_losses)
    total_mae_for_set = mae_losses_stacked.mean()

    mean_nll_losses = torch.stack(mean_nll_losses)
    mean_nll_losses = mean_nll_losses.mean()

    mean_nll_losses = mean_nll_losses.item()
    total_mae_for_set = total_mae_for_set.item()

    return mean_nll_losses, total_mae_for_set


def run_pipeline(
    seq_len: int,
    epochs: int,
    buckets: int,
    steps: int,
    learning_rate: float,
    emsize: int,
    nhead: int,
    nhid: int,
    nlayers: int,
) -> dict:
    dataset_name = "neps_pipeline_model"

    batch_size = 40
    dropout = 0.0
    weight_decay = 0.0

    kinpfn = KINPFN(
        dataset_name=dataset_name,
        seq_len=seq_len,
        epochs=epochs,
        buckets=buckets,
        steps=steps,
        batch_size=batch_size,
        lr=learning_rate,
        emsize=emsize,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        weight_decay=weight_decay,
    )

    trained_model = kinpfn.model

    if trained_model is not None:
        print("Load trained model!")
    else:
        print("No loaded model!")
        trained_model = kinpfn.model
        train_results = kinpfn.train_new_model(
            get_batch_function=get_batch_multi_modal_distribution_prior,
            is_neps_run=True,
        )
        training_error = train_results[0]
        trained_model = train_results[2]

    mean_nll_for_val_set, total_mae_for_set = neps_model_validation_pipeline(
        trained_model=trained_model
    )
    loss = mean_nll_for_val_set
    return loss


pipeline_space = {
    "seq_len": neps.CategoricalParameter([200, 300, 500, 700, 1000, 1400]),
    "epochs": neps.IntegerParameter(250, 3000, is_fidelity=True),
    "buckets": neps.CategoricalParameter([100, 1000, 5000, 10000]),
    "steps": neps.IntegerParameter(50, 100),
    "learning_rate": neps.FloatParameter(1e-5, 1e-3, log=True),
    "emsize": neps.CategoricalParameter([256, 512]),
    "nhead": neps.CategoricalParameter([4, 8]),
    "nhid": neps.CategoricalParameter([512, 1024]),
    "nlayers": neps.CategoricalParameter([2, 3, 4, 6, 8]),
}

if __name__ == "__main__":
    start_time = time.time()

    set_seed(123)
    logging.basicConfig(level=logging.INFO)

    tblogger_status = tblogger.get_status()
    logging.info(f"TB LOGGER STATUS: {tblogger_status}\n")

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=f"./neps_results/neps_synthetic_prior{start_time}",
        max_evaluations_total=150,
        searcher="hyperband",
        post_run_summary=True,
        loss_value_on_error=100,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"NePS Execution time: {execution_time} seconds\n")

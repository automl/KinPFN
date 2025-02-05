import os
import torch
import polars as pl
import numpy as np


from kinpfn.priors import Batch


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
    length_order = []
    file_names_in_order = []
    for subdir, _, files in os.walk(val_set_dir):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(subdir, file)
                data = pl.read_csv(
                    path,
                    has_header=False,
                    columns=[2, 4],
                    n_rows=seq_len,
                )

                folding_times = data["column_3"].to_numpy()
                sequence = data["column_5"][0]
                length_order.append(len(sequence))
                file_names_in_order.append(file)
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
    return Batch(x=x, y=y, target_y=y), length_order, file_names_in_order

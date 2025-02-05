import numpy as np
import torch
from scipy.interpolate import interp1d

from kinpfn.priors import Batch


## Family of multi-modal Gaussian distributions
def multi_modal_distribution(
    x, num_peaks, log_start, log_end, rng=np.random.default_rng(seed=np.random.seed())
):
    theta = rng.uniform(log_start, log_end)
    means = (rng.uniform(log_start + 1, log_end + 1, num_peaks) + theta).astype(
        np.float32
    )
    stds = rng.uniform(0.1, (log_end - log_start) / 5, num_peaks).astype(np.float32)
    distribution = np.zeros_like(x, dtype=np.float32)

    for mean, std in zip(means, stds):
        distribution += np.exp(-((np.log(x) - mean) ** 2) / (2 * std**2))

    return distribution


def sample_from_multi_modal_distribution(batch_size, seq_len, num_features):

    xs = torch.zeros(batch_size, seq_len, num_features, dtype=torch.float32)
    ys = torch.empty(batch_size, seq_len, dtype=torch.float32)

    rng = np.random.default_rng(seed=np.random.seed())
    log_start = -6
    log_end = 15
    x = np.logspace(log_start, log_end, seq_len, dtype=np.float32)

    for i in range(batch_size):
        num_peaks = rng.integers(2, 6)
        own_pdf = multi_modal_distribution(x, num_peaks, log_start, log_end, rng=rng)
        own_pdf /= np.trapz(own_pdf, x)
        own_cdf = np.cumsum(own_pdf).astype(np.float32)
        own_cdf /= own_cdf[-1]
        inverse_cdf = interp1d(
            own_cdf, x, bounds_error=False, fill_value=(x[0], x[-1]), kind="linear"
        )
        uniform_samples = rng.uniform(0, 1, seq_len).astype(np.float32)
        samples = inverse_cdf(uniform_samples).astype(np.float32)
        ys[i] = torch.tensor(samples, dtype=torch.float32)

    return xs, ys


def get_batch_multi_modal_distribution_prior(
    batch_size, seq_len, num_features=1, hyperparameters=None, **kwargs
):
    xs, ys = sample_from_multi_modal_distribution(
        batch_size=batch_size, seq_len=seq_len, num_features=num_features
    )
    # Log encoding y
    ys = torch.log10(ys)

    return Batch(
        x=xs.transpose(0, 1),
        y=ys.transpose(0, 1),
        target_y=ys.transpose(0, 1),
    )

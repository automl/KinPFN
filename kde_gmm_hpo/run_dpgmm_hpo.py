import torch
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import ParameterSampler
from get_data_batch import get_batch_testing_folding_times, get_dataset_size
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_bgmm_with_config(params):
    """
    Runs the Bayesian Gaussian Mixture Model (BGMM) with given parameters and evaluates using NLL.
    """
    # Load data (replace with actual path to your dataset)
    val_set_dir = "../neps_validation_set"
    seq_len = 1000
    training_points = 25

    # Fetch batch data
    batch, _, _ = get_batch_testing_folding_times(
        val_set_dir=val_set_dir, seq_len=seq_len
    )
    dataset_size = get_dataset_size(val_set_dir)

    indices = list(range(dataset_size))

    x = batch.x
    y_folding_times = batch.y

    nll_scores = []

    # Collect train and test data from all batches
    for i in indices:
        batch_index = i
        train_indices = torch.randperm(seq_len)[:training_points]

        # Prepare train and test data
        train_y_folding_times = y_folding_times[train_indices, batch_index]
        test_y_folding_times = y_folding_times[:, batch_index]

        train_y_folding_times = train_y_folding_times.reshape(-1, 1)
        test_y_folding_times = test_y_folding_times.reshape(-1, 1)

        # Fit Bayesian GMM model
        bgmm = BayesianGaussianMixture(
            **params, n_components=5, max_iter=1000, random_state=42
        )
        bgmm.fit(train_y_folding_times)
        nll = -bgmm.score(test_y_folding_times)
        nll_scores.append(nll)

    mean_nll = np.mean(nll_scores)
    print(f"Mean NLL (BGMM): {mean_nll}")
    return mean_nll


# Set up hyperparameter grid for Bayesian GMM
# param_dist_bgmm = {
#     "n_components": np.arange(1, 10),
#     "covariance_type": ["full", "tied", "diag", "spherical"],
#     "tol": [1e-3, 1e-4, 1e-5],
#     "weight_concentration_prior": [1e-2, 1e-3, 1e-4],
# }
param_dist_bgmm = {
    "weight_concentration_prior": np.logspace(-4, 2, 1000),
}

param_sampler_bgmm = list(
    ParameterSampler(param_dist_bgmm, n_iter=1000, random_state=42)
)

# Perform custom cross-validation with hyperparameter search for Bayesian GMM
best_params_bgmm, best_score_bgmm = None, float("inf")

# Logging results
f = open("bgmm_hpo_results.csv", "w")
f.write("weight_concentration_prior,nll\n")

for params in param_sampler_bgmm:
    print(f"Evaluating Bayesian GMM with params: {params}")

    # Evaluate the model with the current hyperparameters
    score = run_bgmm_with_config(params)

    # Write the results to the log file
    f.write(f"{params['weight_concentration_prior']},{score}\n")

    # Update the best parameters if the current model is better
    if score < best_score_bgmm:
        best_score_bgmm = score
        best_params_bgmm = params

# Output the best parameters and the best score
print(f"Best Parameters for Bayesian GMM: {best_params_bgmm}")
print(f"Best NLL Score for Bayesian GMM: {best_score_bgmm}")

# Write the best parameters and score to the log file
f.write("\n")
f.write("Best Parameters\n")
f.write(f"Params: {best_params_bgmm}\n")
f.write(f"Best NLL Score: {best_score_bgmm}\n")

# Close the log file
f.close()

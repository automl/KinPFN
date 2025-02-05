from kinpfn.model import KINPFN
from kinpfn.train_kinpfn import train_kinpfn
from kinpfn.prior_kinpfn import get_batch_multi_modal_distribution_prior


def test_kinpfn_imports():
    assert KINPFN is not None, "KINPFN class should be imported successfully"
    assert (
        train_kinpfn is not None
    ), "train_kinpfn function should be imported successfully"
    assert (
        get_batch_multi_modal_distribution_prior is not None
    ), "get_batch_multi_modal_distribution_prior function should be imported successfully"

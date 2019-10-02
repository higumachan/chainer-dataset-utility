import pytest
from chainer.dataset import DatasetMixin

from chainerds.dataset.through.tansform import ThroughMethodTransform
from chainerds.dataset.through.through_method_dataset import sample_wise_method


class DS(DatasetMixin):
    def __len__(self):
        return 10

    def get_example(self, i):
        return i

    @sample_wise_method
    def get_2times(self, i):
        return i * 2

    def no_sample_wise_2times(self, i):
        return i * 2


def _3times(x):
    return x * 3


def test_transform():
    ds = DS()
    _3times_ds = ThroughMethodTransform(ds, _3times)
    assert _3times_ds.get_example(3) == 9
    assert _3times_ds.get_2times(3) == 6


def test_no_sample_wise_method():
    ds = DS()
    _3times_ds = ThroughMethodTransform(ds, _3times)
    with pytest.raises(ValueError):
        _3times_ds.no_sample_wise_2times(3)

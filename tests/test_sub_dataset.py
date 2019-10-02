import pytest
from chainer.dataset import DatasetMixin

from chainerds.dataset.through.sub_dataset import ThroughMethodSubDataset
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


def test_sub_dataset():
    ds = DS()
    subds = ThroughMethodSubDataset(ds, 3, 10)
    assert subds.get_example(0) == 3
    assert subds.get_2times(0) == 6


def test_no_sample_wise_method():
    ds = DS()
    subds = ThroughMethodSubDataset(ds, 3, 10)
    with pytest.raises(ValueError):
        subds.no_sample_wise_2times(3)

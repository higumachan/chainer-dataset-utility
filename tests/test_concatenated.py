import pytest
from chainer.dataset import DatasetMixin

from chainerds.dataset.through.concatenated_dataset import ThroughMethodConcatenatedDataset
from chainerds.dataset.through.through_method_dataset import sample_wise_method


class DS1(DatasetMixin):
    def __len__(self):
        return 10

    def get_example(self, i):
        return i

    @sample_wise_method
    def get_2times(self, i):
        return i * 2

    def no_sample_wise_2times(self, i):
        return i * 2


class DS2(DatasetMixin):
    def __len__(self):
        return 10

    def get_example(self, i):
        return i * 3

    @sample_wise_method
    def get_2times(self, i):
        return i * 2

    def no_sample_wise_2times(self, i):
        return i * 2


def test_concatenated():
    ds1 = DS1()
    ds2 = DS2()
    cat_ds = ThroughMethodConcatenatedDataset(ds1, ds2)
    assert len(cat_ds) == 20
    assert cat_ds[11] == 3
    assert cat_ds.get_2times(11) == 2


def test_no_sample_wise_method():
    ds1 = DS1()
    ds2 = DS2()
    cat_ds = ThroughMethodConcatenatedDataset(ds1, ds2)
    with pytest.raises(ValueError):
        cat_ds.no_sample_wise_2times(3)
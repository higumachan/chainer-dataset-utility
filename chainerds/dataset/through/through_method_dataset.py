from functools import wraps

from chainer.dataset import DatasetMixin
from chainer.datasets import TransformDataset, SubDataset


class ThroughMethodDatasetMixin(DatasetMixin):
    @property
    def parent_dataset(self):
        raise NotImplemented

    def __getattr__(self, item):
        method = getattr(self.parent_dataset, item)

        def index_change(i):
            return method(self._index_mapping(i))

        if getattr(method, "_sample_wise_method_mark", False):
            return wraps(method)(index_change)
        raise ValueError("this method is not sample_wise_method")

    def _index_mapping(self, i):
        return i


def sample_wise_method(func):
    func._sample_wise_method_mark = True
    return func


class ThroughMethodTransform(ThroughMethodDatasetMixin):
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    @property
    def parent_dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        in_data = self._dataset[i]
        return self._transform(in_data)


class ThroughMethodSubDataset(ThroughMethodDatasetMixin):
    def __init__(self, dataset, start, finish, order=None):
        if start < 0 or finish > len(dataset):
            raise ValueError('subset overruns the base dataset.')
        self._dataset = dataset
        self._start = start
        self._finish = finish
        self._size = finish - start
        if order is not None and len(order) != len(dataset):
            msg = ('order option must have the same length as the base '
                   'dataset: len(order) = {} while len(dataset) = {}'.format(
                       len(order), len(dataset)))
            raise ValueError(msg)
        self._order = order

    def __len__(self):
        return self._size

    def get_example(self, i):
        return self._dataset[self._index_mapping(i)]

    def _index_mapping(self, i):
        if i >= 0:
            if i >= self._size:
                raise IndexError('dataset index out of range')
            index = self._start + i
        else:
            if i < -self._size:
                raise IndexError('dataset index out of range')
            index = self._finish + i

        if self._order is not None:
            index = self._order[index]
        return index

    @property
    def parent_dataset(self):
        return self._dataset


if __name__ == '__main__':
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

    ds = DS()
    _3times_ds = ThroughMethodTransform(ds, _3times)
    subds = ThroughMethodSubDataset(ds, 3, 10)

    assert _3times_ds.get_example(3) == 9
    assert _3times_ds.get_2times(3) == 6
    assert subds.get_example(0) == 3
    assert subds.get_2times(0) == 6
    try:
        _3times_ds.no_sample_wise_2times(3)
        assert False
    except ValueError:
        pass

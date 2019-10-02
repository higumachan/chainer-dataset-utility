from functools import wraps

from chainer.dataset import DatasetMixin


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

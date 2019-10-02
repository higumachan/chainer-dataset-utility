from functools import wraps

from chainer.dataset import DatasetMixin


class ThroughMethodDatasetMixin(DatasetMixin):
    def __getattr__(self, item):
        def index_change(i):
            parent_dataset, index = self._dataset_index_mapping(i)
            method = getattr(parent_dataset, item)
            if getattr(method, "_sample_wise_method_mark", False):
                return method(index)
            raise ValueError("this method is not sample_wise_method")
        return index_change

    def _dataset_index_mapping(self, i):
        raise NotImplementedError()


def sample_wise_method(func):
    func._sample_wise_method_mark = True
    return func

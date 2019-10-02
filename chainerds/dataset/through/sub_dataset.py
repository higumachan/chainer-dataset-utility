from chainerds.dataset.through.through_method_dataset import ThroughMethodDatasetMixin


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
        return self._dataset[self._dataset_index_mapping(i)[1]]

    def _dataset_index_mapping(self, i):
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
        return self._dataset, index

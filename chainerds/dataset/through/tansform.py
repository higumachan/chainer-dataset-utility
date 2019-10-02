from chainerds.dataset.through.through_method_dataset import ThroughMethodDatasetMixin


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
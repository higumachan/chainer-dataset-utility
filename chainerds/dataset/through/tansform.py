from chainerds.dataset.through.through_method_dataset import ThroughMethodDatasetMixin


class ThroughMethodTransform(ThroughMethodDatasetMixin):
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        in_data = self._dataset[i]
        return self._transform(in_data)

    def _dataset_index_mapping(self, i):
        return self._dataset, i

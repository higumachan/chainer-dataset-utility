from chainer.dataset import DatasetMixin
import time


class MeasureTime(DatasetMixin):
    def __init__(self, super_dataset: DatasetMixin, name: str):
        self.super_dataset = super_dataset
        self.name = name

    def get_example(self, i):
        s = time.time()
        example = self.super_dataset.get_example(i)
        e = time.time()

        print(self.name, e - s)  # TODO(higumachan): ほんとはReporterに飛ばす
        return example

    def __len__(self):
        return len(self.super_dataset)

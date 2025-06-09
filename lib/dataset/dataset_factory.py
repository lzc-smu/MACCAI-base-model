from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pat import Dataset
from .task2 import Dataset2

dataset_factory = {
  'pat': Dataset,
  'task2': Dataset2,
}


def get_dataset(dataset):
  return dataset_factory[dataset]

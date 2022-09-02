from typing import Generator, List

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mergernet.core.constants import RANDOM_SEED


class DistributionKFold:
  def __init__(self, bins: int):
    self.bins = bins


  def split(self, distribution) -> Generator:
    bin_edges = np.histogram_bin_edges(distribution, bins=self.bins)

    for i in range(len(bin_edges) - 1):
      bin_ids = np.asarray(
        (distribution >= bin_edges[i]) & (distribution <= bin_edges[i + 1])
      ).nonzero()[0]
      yield bin_ids



class StratifiedDistributionKFold:
  def __init__(self, distribution, n_splits: int, bins: int, shuffle: bool = True):
    self.n_splits = n_splits
    self.bins = bins
    self.distribution = distribution
    self.shuffle = shuffle
    self.random_seed = RANDOM_SEED if self.shuffle else None


  @staticmethod
  def compute_max_bins(y, distribution: List[float], n_classes: int, n_splits: int) -> int:
    continue_testing = True
    max_bins = 1
    y = np.array(y)
    distribution = np.array(distribution)

    while continue_testing:
      dist_kf = DistributionKFold(bins=max_bins)

      for bin_ids in dist_kf.split(distribution=distribution):
        y_bin = y[bin_ids]
        _, counts = np.unique(y_bin, return_counts=True)
        # class_counts = dict(zip(unique, counts))

        # check ocurrences for all classes
        continue_testing = continue_testing and (len(counts) == n_classes)

        # check ocurrences for at least 1 example of each class in each fold
        for c in counts:
          continue_testing = continue_testing and (c >= n_classes * n_splits * max_bins)

        max_bins += 1 if continue_testing else 0

    return max_bins


  def split_ids(self, X, y) -> dict:
    train = [np.array([], dtype=np.int32) for _ in range(self.n_splits)]
    test = [np.array([], dtype=np.int32) for _ in range(self.n_splits)]

    dist_kf = DistributionKFold(bins=self.bins)
    class_kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_seed)

    # First split: distribution split
    for bin_ids in dist_kf.split(distribution=self.distribution):
      y_bin = y[bin_ids]

      # Second split: class split
      for i, (train_ids, test_ids) in enumerate(class_kf.split(bin_ids, y_bin)):
        train[i] = np.concatenate((train[i], bin_ids[train_ids].astype(np.int32)))
        test[i] = np.concatenate((test[i], bin_ids[test_ids].astype(np.int32)))

    return (train, test)


  def split_all(self, X, y) -> dict:
    X_train = [[] for _ in range(self.n_splits)]
    X_test = [[] for _ in range(self.n_splits)]
    y_train = [[] for _ in range(self.n_splits)]
    y_test = [[] for _ in range(self.n_splits)]

    dist_kf = DistributionKFold(bins=self.bins)
    class_kf = StratifiedKFold(n_splits=self.n_splits)

    # First split: distribution split
    for bin_ids in dist_kf.split(distribution=self.distribution):
      X_bin = X[bin_ids]
      y_bin = y[bin_ids]

      # Second split: class split
      for i, (train_ids, test_ids) in enumerate(class_kf.split(X_bin, y_bin)):
        X_train[i] = np.concatenate((X_train[i], X_bin[train_ids]))
        X_test[i] = np.concatenate((X_test[i], X_bin[test_ids]))
        y_train[i] = np.concatenate((y_train[i], y_bin[train_ids]))
        y_test[i] = np.concatenate((y_test[i], y_bin[test_ids]))

    # Notes:
    # len(X_train) == n_splits, each elements of the list represents a fold
    # each

    return {
      'X_train': X_train,
      'X_test': X_test,
      'y_train': y_train,
      'y_test': y_test
    }

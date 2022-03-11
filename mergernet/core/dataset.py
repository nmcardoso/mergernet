"""Dataset Module: high-level abstraction of dataset.

This module defines the rules for loading a previously generated ``.tfrecord`` dataset.

The main class is :py:class:`Dataset`, which is a high-level representation of the
dataset and can be used for all development process of this deep learning model.

This module defines others classes and functions as well, who perform complementary tasks.
"""


from pathlib import Path
from typing import Dict, List, Generator, Sequence, Tuple, Union

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
import pandas as pd

from mergernet.core.utils import load_image, load_table
from mergernet.core.constants import RANDOM_SEED
from mergernet.services.google import GDrive




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



class DatasetConfig:
  """Configuration params for dataset."""
  def __init__(
    self,
    name: str = None,
    archive_url: str = None,
    table_url: str = None,
    archive_path: Path = None,
    images_path: Path = None,
    table_path: Path = None,
    X_column: str = 'ID',
    y_column: str = 'class',
    fold_column: str = 'fold',
    X_column_suffix: str = '', # filename extension
    detect_img_extension: bool = False,
    label_map: dict = None,
    image_shape: tuple = None,
    n_classes: int = None
  ):
    self.name = name
    self.archive_url = archive_url
    self.table_url = table_url
    self.archive_path = archive_path
    self.images_path = images_path
    self.table_path = table_path
    self.X_column = X_column
    self.y_column = y_column
    self.fold_column = fold_column
    self.X_column_suffix = X_column_suffix
    self.detect_img_extension = detect_img_extension
    self.label_map = label_map
    self.image_shape = image_shape
    self.n_classes = n_classes

  def __repr__(self) -> str:
    return (
      f'Using dataset {self.name.upper()}.\n'
      f'Archive path:\t\t{str(self.archive_path.resolve())}'
      f'Images path:\t\t{str(self.images_path.resolve())}'
      f'Table path:\t\t{str(self.table_path.resolve())}'
      f'Label map:\t\t{str(self.label_map)}'
    )



DARG_NO_INSPECTION = DatasetConfig(
  # table_url=GDrive.get_url('1yHnyOdXS-HKzIsbenSi646jyf2AWU9vo'),
  name='darg_no_inspection',
  archive_url=GDrive.get_url('1ltKXhZgA4Ab60FGKCqybhebSiAL2LMgG'),
  table_url=GDrive.get_url('1QgUYkzcjaCmo-kcgM8s2U8PYlx0C58oD'),
  archive_path=Path('sdss_lupton_jpg_128.tar.xz'),
  images_path=Path('sdss_lupton_jpg_128'),
  table_path=Path('reference_darg.csv'),
  X_column='filename',
  y_column='class',
  fold_column='fold',
  label_map={ 'E': 0, 'M': 1, 'S': 2 },
  image_shape=(128, 128, 3),
  n_classes=3
)
"""Default configuration object for RGB dataset."""

MESD_SDSS_128 = DatasetConfig(
  name='mesd_sdss_128',
  archive_url=GDrive.get_url('1YZ7A9rVglf5NJp27rW8-6aJXBfBNPfYv'),
  table_url=GDrive.get_url('1uaRXWUskBrLHi-IGLhZ5T6yIa8w5hi4H'),
  archive_path=Path('mesd_sdss_128.tar.xz'),
  images_path=Path('mesd_sdss_128'),
  table_path=Path('mesd.csv'),
  X_column='iauname',
  X_column_suffix='.jpg',
  y_column='class',
  fold_column='fold',
  label_map={'merger': 0, 'elliptical': 1, 'spiral': 2, 'disturbed': 3},
  image_shape=(128, 128, 3),
  n_classes=4
)
"""MESD dataset with SDSS 128x128 images."""

MESD_LEGACY_128 = DatasetConfig(
  name='mesd_legacy_128',
  archive_url=GDrive.get_url('1cTU0SVEv3qVeVxF7pzhtOP7S-bx5cPOP'),
  table_url=GDrive.get_url('1uaRXWUskBrLHi-IGLhZ5T6yIa8w5hi4H'),
  archive_path=Path('mesd_legacy_128.tar.xz'),
  images_path=Path('mesd_legacy_128'),
  table_path=Path('mesd.csv'),
  X_column='iauname',
  X_column_suffix='.jpg',
  y_column='class',
  fold_column='fold',
  label_map={'merger': 0, 'elliptical': 1, 'spiral': 2, 'disturbed': 3},
  image_shape=(128, 128, 3),
  n_classes=4
)
"""MESD dataset with Legacy Survey 128x128 images."""

DATASET_REGISTRY = {
  'darg_no_inspection': DARG_NO_INSPECTION,
  'mesd_sdss_128': MESD_SDSS_128,
  'mesd_legacy_128': MESD_LEGACY_128
}



class Dataset:
  """High-level representation of dataset.

  Parameters
  ----------
  ds_split: str
    Split path.

  Attributes
  ----------
  config: DatasetConfig
    Configuration object.
  """

  def __init__(
    self,
    data_path: Union[str, Path] = Path(''),
    ds: str = '',
    in_memory: bool = False
  ):
    self.ds = ds
    self.data_path = Path(data_path)
    self.in_memory = in_memory

    self.config = DATASET_REGISTRY.get(self.ds.lower(), MESD_SDSS_128)
    self.config.archive_path = data_path / self.config.archive_path
    self.config.images_path = data_path / self.config.images_path
    self.config.table_path = data_path / self.config.table_path

    if not self.is_dataset_downloaded():
      self.download()

    if self.config.detect_img_extension:
      self._detect_img_extension()


  def _detect_img_extension(self):
    self.config.X_column_suffix = next(self.config.images_path.iterdir()).suffix


  def _discretize_label(self, y: np.ndarray) -> np.ndarray:
    y_int = np.empty(y.shape, dtype=np.int64)

    for k, v in self.config.label_map.items():
      y_int[y == k] = v

    return y_int


  def is_dataset_downloaded(self):
    return self.config.images_path.is_dir() and self.config.table_path.is_file()


  def download(self) -> None:
    """Check if destination path exists, create missing folders and download
    the dataset files from web resource for a specified dataset type.
    """
    if not self.config.archive_path.parent.exists():
      self.config.archive_path.parent.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.get_file(
      fname=self.config.archive_path.resolve(),
      origin=self.config.archive_url,
      cache_subdir=self.config.archive_path.parent.resolve(),
      archive_format='tar',
      extract=True
    )

    if not self.config.table_path.parent.exists():
      self.config.table_path.parent.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.get_file(
      fname=self.config.table_path.resolve(),
      origin=self.config.table_url,
      cache_subdir=self.config.table_path.parent.resolve()
    )


  def get_fold(self, fold: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    df = pd.read_csv(self.config.table_path)

    df_test = df[df[self.config.fold_column] == fold]
    df_train = df[df[self.config.fold_column] != fold]

    X_train = df_train[self.config.X_column].to_numpy()
    y_train = df_train[self.config.y_column].to_numpy()
    X_test = df_test[self.config.X_column].to_numpy()
    y_test = df_test[self.config.y_column].to_numpy()

    X_train = np.array([
      str((self.config.images_path / (X + self.config.X_column_suffix)).resolve())
      for X in X_train
    ])

    X_test = np.array([
      str((self.config.images_path / (X + self.config.X_column_suffix)).resolve())
      for X in X_test
    ])

    if self.in_memory:
      X_train = [load_image(path) for path in X_train]
      X_test = [load_image(path) for path in X_test]

    if self.config.label_map:
      y_train = self._discretize_label(y_train)
      y_test = self._discretize_label(y_test)

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return ds_train, ds_test


  def get_X_by_fold(self, fold: int, kind='test') -> np.ndarray:
    df = pd.read_csv(self.config.table_path)

    if kind == 'test':
      df = df[df[self.config.fold_column] == fold]
    else:
      df = df[df[self.config.fold_column] != fold]

    return df[self.config.X_column].to_numpy()


  def get_X(self) -> np.ndarray:
    df = pd.read_csv(self.config.table_path)
    return df[self.config.X_column].to_numpy()


  def get_preds_dataset(self) -> tf.data.Dataset:
    df = pd.read_csv(self.config.table_path)

    X = np.array([
      str((self.config.images_path / (_X + self.config.X_column_suffix)).resolve())
      for _X in df[self.config.X_column].to_numpy()
    ])

    return tf.data.Dataset.from_tensor_slices(X)


  def compute_class_weight(self) -> dict:
    y = pd.read_csv(self.config.table_path)[self.config.y_column].to_numpy()
    if self.config.label_map:
      y = self._discretize_label(y)

    classes, cardinalities = np.unique(y, return_counts=True)
    total = np.sum(cardinalities)
    n = len(cardinalities)

    weight_map = {}
    for class_, cardinality in zip(classes, cardinalities):
      weight_map[class_] = total / (n * cardinality)

    return weight_map


  @staticmethod
  def concat_fold_column(
    df: pd.DataFrame,
    fname_column: str = None,
    class_column: str = None,
    r_column: str = None,
    n_splits: int = 5,
    bins: int = 3
  ) -> pd.DataFrame:
    df = df.copy()
    X = df[fname_column].to_numpy() # filenames
    y = df[class_column].to_numpy() # class labels
    r = df[r_column].to_numpy() # r band magnitude

    kf = StratifiedDistributionKFold(distribution=r, n_splits=n_splits, bins=bins)
    _, test_ids = kf.split_ids(X, y) # using test ids to identify folds

    folds = np.empty(X.shape, dtype=np.int32)

    for i in range(len(test_ids)):
      folds[test_ids[i]] = i

    df['fold'] = folds

    return df


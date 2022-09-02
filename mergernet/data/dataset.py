"""
Dataset Module: high-level abstraction of dataset.

This module defines the rules for loading a previously generated ``.tfrecord`` dataset.

The main class is :py:class:`Dataset`, which is a high-level representation of the
dataset and can be used for all development process of this deep learning model.

This module defines others classes and functions as well, who perform complementary tasks.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from mergernet.core.experiment import Experiment
from mergernet.core.utils import load_image
from mergernet.data.dataset_config import DatasetConfig, DatasetRegistry
from mergernet.data.kfold import StratifiedDistributionKFold
from mergernet.data.preprocessing import load_jpg, load_png, one_hot_factory

L = logging.getLogger(__name__)



class Dataset:
  """
  High-level representation of the dataset. This class abstracts all IO
  operations of the dataset (e.g. download, prepare, split)

  Attributes
  ----------
  registry: DatasetRegistry
    A registry containing all datasets configurations

  Parameters
  ----------
  config: DatasetConfig
    The configuration object of the database get from `Dataset.registry`
    attribute
  """
  registry = DatasetRegistry()

  def __init__(
    self,
    config: DatasetConfig,
    in_memory: bool = False
  ):
    self.in_memory = in_memory

    data_path = Path(Experiment.local_shared_path)
    self.config = config
    self.config.archive_path = data_path / self.config.archive_path
    self.config.images_path = data_path / self.config.images_path
    self.config.table_path = data_path / self.config.table_path

    if not self.is_dataset_downloaded():
      self.download()

    if self.config.detect_img_extension:
      self._detect_img_extension()


  def _detect_img_extension(self):
    """
    Set X_column_suffix attribute of DatasetConfig to detected image extension
    """
    self.config.X_column_suffix = next(self.config.images_path.iterdir()).suffix


  def _discretize_label(self, y: np.ndarray) -> np.ndarray:
    """
    Find all ocurrences in table that matches ``DatasetConfig.label_map`` key
    and replaces with respective value.

    Parameters
    ----------
    y: np.ndarray
    """
    y_int = np.empty(y.shape, dtype=np.int64)

    for k, v in self.config.label_map.items():
      y_int[y == k] = v

    return y_int


  def is_dataset_downloaded(self) -> bool:
    """
    Check if dataset files are downloaded locally at
    ``Experiment.local_shared_path``

    Returns
    -------
    bool
      True if the images dir and the table are found, False otherwise
    """
    return self.config.images_path.is_dir() and self.config.table_path.is_file()


  def download(self):
    """
    Check if destination path exists, create missing folders and download
    the dataset files from web resource for a specified dataset type.
    """
    # Download images
    if not self.config.archive_path.parent.exists():
      self.config.archive_path.parent.mkdir(parents=True, exist_ok=True)

    if type(self.config.archive_url) == dict:
      archive_url = self.config.archive_url.get(
        'sciserver', self.config.archive_url.get('gdrive')
      )
    else:
      archive_url = self.config.archive_url

    tf.keras.utils.get_file(
      fname=self.config.archive_path.resolve(),
      origin=archive_url,
      cache_subdir=self.config.archive_path.parent.resolve(),
      archive_format='tar',
      extract=True
    )

    # Download table
    if not self.config.table_path.parent.exists():
      self.config.table_path.parent.mkdir(parents=True, exist_ok=True)

    if type(self.config.table_url) == dict:
      table_url = self.config.table_url.get(
        'sciserver', self.config.table_url.get('gdrive')
      )
    else:
      table_url = self.config.table_url

    tf.keras.utils.get_file(
      fname=self.config.table_path.resolve(),
      origin=table_url,
      cache_subdir=self.config.table_path.parent.resolve()
    )


  def get_fold(self, fold: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Generates the train and test dataset based on selected fold

    Parameters
    ----------
    fold: int
      The fold which will be used as test, the all other folds will be used as
      train

    Retruns
    -------
    tuple of tf.data.Dataset
      A tuple containing two datasets, the first is the train dataset and the
      secound is the test dataset
    """
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


  def prepare_data(
    self,
    ds: tf.data.Dataset,
    batch_size: int = 64,
    buffer_size: int = 1000,
    kind='train'
  ):
    if self.config.X_column_suffix == '.jpg':
      ds = ds.map(load_jpg)
      L.info('Apply: load_jpg')
    elif self.config.X_column_suffix == '.png':
      ds = ds.map(load_png)
      L.info('Apply: load_png')

    if kind == 'train':
      ds = ds.map(one_hot_factory(self.config.n_classes))
      L.info('Apply: one_hot')

    ds = ds.cache()
    L.info('Apply: cache')

    ds = ds.shuffle(buffer_size)
    L.info('Apply: shuffle')

    ds = ds.batch(batch_size)
    L.info('Apply: batch')

    ds = ds.prefetch(tf.data.AUTOTUNE)
    L.info('Apply: prefetch')

    return ds


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


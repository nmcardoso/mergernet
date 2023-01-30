"""
Dataset Module: high-level abstraction of dataset.

This module defines the rules for loading a previously generated ``.tfrecord`` dataset.

The main class is :py:class:`Dataset`, which is a high-level representation of the
dataset and can be used for all development process of this deep learning model.

This module defines others classes and functions as well, who perform complementary tasks.
"""

import logging
from pathlib import Path
from shutil import copy2
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname, iauname_relative_path, load_image
from mergernet.data.dataset_config import (DatasetConfig, DatasetRegistry,
                                           GoogleDriveResource, HTTPResource)
from mergernet.data.kfold import StratifiedDistributionKFold
from mergernet.data.preprocessing import load_jpg, load_png, one_hot_factory

L = logging.getLogger(__name__)



class Dataset:
  """
  High-level representation of the dataset. This class abstracts all IO
  operations of the dataset (e.g. download, prepare, split)

  Parameters
  ----------
  config: DatasetConfig
    The configuration object of the database get from `Dataset.registry`
    attribute
  """
  registry: DatasetRegistry = DatasetRegistry()
  """A registry containing all datasets configurations"""

  def __init__(
    self,
    config: DatasetConfig,
    in_memory: bool = False
  ):
    self.in_memory = in_memory
    self.config = config
    self._weight_map = None

    # setup paths
    data_path = Experiment.local_shared_path
    if config.archive_path:
      self.config.archive_path = data_path / self.config.archive_path
    if config.images_path:
      self.config.images_path = data_path / self.config.images_path
    if config.table_path:
      self.config.table_path = data_path / self.config.table_path

    # cast positions to numpy array
    if config.positions is not None:
      self.config.positions = np.array(self.config.positions)

    # create table for unregistered datasets
    self._create_dataset_table()

    # download dataset data
    # if not self.is_dataset_downloaded():
    self.download()

    if self.config.image_transform is not None:
      self._transform_images()


  def _discretize_label(self, y: np.ndarray) -> np.ndarray:
    """
    Find all ocurrences in table that matches ``DatasetConfig.label_map`` key
    and replaces with respective value.

    Parameters
    ----------
    y: np.ndarray
    """
    y_int = np.empty(y.shape, dtype=np.uint8)

    for class_index, class_name in enumerate(self.config.labels):
      y_int[y == class_name] = class_index

    return y_int


  def _create_dataset_table(self):
    """
    Scan the images table and create a csv table with filenames if the
    dataset config has no table.
    """
    if self.config.table_url is None:
      self.config.table_path = Experiment.local_shared_path / f'{self.config.name}.csv'
      self.config.image_column = 'iauname'

      if not self.config.table_path.exists():
        images = list(self.config.images_path.glob(f'**/*.{self.config.image_extension}'))

        if self.config.positions is not None and len(images) == 0:
          ra = self.config.positions[:, 0]
          dec = self.config.positions[:, 1]
          iaunames = iauname(ra, dec)
          df_data = {
            'ra': ra,
            'dec': dec,
            'iauname': iaunames
          }
        else:
          iaunames = [p.stem for p in images]
          df_data = {'iauname': iaunames}

        pd.DataFrame(df_data).to_csv(self.config.table_path, index=False)


  def _transform_images(self):
    iaunames = self.get_X()

    if self.config.image_service is not None:
      suffix = self.config.image_service.image_format
    else:
      suffix = self.config.image_extension

    images = iauname_relative_path(
      iaunames=iaunames,
      prefix=self.config.images_path,
      suffix=f'.{suffix}'
    )
    save_paths = self.get_images_paths(iaunames)

    self.config.image_transform.batch_transform(images, save_paths)


  def is_dataset_downloaded(self) -> bool:
    """
    Check if dataset files are downloaded locally at
    ``Experiment.local_shared_path``

    Returns
    -------
    bool
      True if the images dir and the table are found, False otherwise
    """
    return self.config.images_path and self.config.images_path.is_dir() and \
           self.config.table_path and self.config.table_path.is_file()


  def download(self):
    """
    Check if destination path exists, create missing folders and download
    the dataset files from web resource for a specified dataset type.
    """
    # Download images
    if self.config.archive_url:
      for i, archive_url in enumerate(self.config.archive_url):
        if isinstance(archive_url, (HTTPResource, str)):
          try:
            tf.keras.utils.get_file(
              fname=self.config.archive_path.resolve(),
              origin=archive_url.url,
              cache_subdir=self.config.archive_path.parent.resolve(),
              archive_format='tar',
              extract=True
            )
            break
          except:
            if i == len(self.config.archive_url) - 1:
              raise RuntimeError("Can't download images archive")
        elif isinstance(archive_url, GoogleDriveResource):
          copy2(archive_url.path, self.config.archive_path)


    # Download table
    if self.config.table_url:
      for i, table_url in enumerate(self.config.table_url):
        try:
          tf.keras.utils.get_file(
            fname=self.config.table_path.resolve(),
            origin=table_url,
            cache_subdir=self.config.table_path.parent.resolve()
          )
          break
        except:
          if i == len(self.config.table_url) - 1:
            raise RuntimeError("Can't download table")

    # Download image from positions
    if self.config.positions is not None and self.config.image_service is not None:
      svc = self.config.image_service
      pos = self.config.positions
      save_paths = iauname_relative_path(
        iaunames=self.get_X(),
        prefix=self.config.images_path,
        suffix=f'.{svc.image_format}'
      )
      _, error = svc.batch_cutout(ra=pos[:, 0], dec=pos[:, 1], save_path=save_paths)
      self.config.image_nested = True

      if len(error) > 0:
        err_iauname = [p.stem for p in error]
        df = self.get_table()
        df = df[~df.iauname.isin(err_iauname)]
        df = df.drop_duplicates(subset=['iauname'])
        df.to_csv(self.config.table_path, index=False)


  def get_fold(self, fold: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Generates the train and test dataset based on selected fold

    Parameters
    ----------
    fold: int
      The fold which will be used as test, the all other folds will be used as
      train

    Returns
    -------
    tuple, tf.data.Dataset
      A tuple containing two datasets, the first is the train dataset and the
      secound is the test dataset
    """
    df = pd.read_csv(self.config.table_path)

    df_test = df[df[self.config.fold_column] == fold]
    df_train = df[df[self.config.fold_column] != fold]

    X_train = df_train[self.config.image_column].to_numpy()
    y_train = df_train[self.config.label_column].to_numpy()
    X_test = df_test[self.config.image_column].to_numpy()
    y_test = df_test[self.config.label_column].to_numpy()

    X_train = np.array([
      str(path.resolve())
      for path in self.get_images_paths(X_train)
    ])

    X_test = np.array([
      str(path.resolve())
      for path in self.get_images_paths(X_test)
    ])

    if self.in_memory:
      X_train = [load_image(path) for path in X_train]
      X_test = [load_image(path) for path in X_test]

    if self.config.labels:
      y_train = self._discretize_label(y_train)
      y_test = self._discretize_label(y_test)

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return ds_train, ds_test


  def get_X_by_fold(self, fold: int, kind='test') -> np.ndarray:
    """
    Get X by fold

    Parameters
    ----------
    fold : int
      Fold number
    kind : str, optional
      one of: 'train' or 'test', by default 'test'

    Returns
    -------
    numpy.ndarray
      X values
    """
    df = pd.read_csv(self.config.table_path)

    if kind == 'test':
      df = df[df[self.config.fold_column] == fold]
    else:
      df = df[df[self.config.fold_column] != fold]

    return df[self.config.image_column].to_numpy()


  def get_X(self) -> np.ndarray:
    df = pd.read_csv(self.config.table_path)
    return df[self.config.image_column].to_numpy()


  def get_table(self) -> pd.DataFrame:
    return pd.read_csv(self.config.table_path)


  def get_images_paths(self, iaunames: List[str]) -> List[Path]:
    if self.config.image_nested:
      return iauname_relative_path(
        iaunames=iaunames,
        prefix=self.config.images_path,
        suffix=f'.{self.config.image_extension}'
      )
    else:
      return [
        self.config.images_path / f'{x}.{self.config.image_extension}'
        for x in iaunames
      ]


  def get_preds_dataset(self) -> tf.data.Dataset:
    iaunames = self.get_X()
    paths = self.get_images_paths(iaunames)

    X = np.array([str(path.resolve()) for path in paths])

    return tf.data.Dataset.from_tensor_slices(X)


  def compute_class_weight(self) -> dict:
    if self._weight_map is not None:
      return self._weight_map

    y = pd.read_csv(self.config.table_path)[self.config.label_column].to_numpy()
    if self.config.labels:
      y = self._discretize_label(y)

    classes, cardinalities = np.unique(y, return_counts=True)
    total = np.sum(cardinalities)
    n = len(cardinalities)

    weight_map = {}
    for class_, cardinality in zip(classes, cardinalities):
      weight_map[class_] = total / (n * cardinality)

    self._weight_map = weight_map

    return weight_map


  def prepare_data(
    self,
    ds: tf.data.Dataset,
    batch_size: int = 64,
    buffer_size: int = 1000,
    kind='train'
  ):
    if self.config.image_extension == 'jpg':
      ds = ds.map(load_jpg)
      L.info('Apply: load_jpg')
    elif self.config.image_extension == 'png':
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



class TFRDataset:
  # https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
  def __init__(
    self,
    save_path: Union[str, Path],
    examples_per_shard: int = 1024,
    train: bool = False
  ):
    self.save_path = Path(save_path)
    self.examples_per_shard = examples_per_shard
    self.train = train

  def create(self, ):
    pass

  def _create_shard(self):
    pass

  def _serialize_array(self, array):
    pass

  def _bytes_feature(self, value):
    pass

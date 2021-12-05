"""Dataset Module: high-level abstraction of dataset.

This module defines the rules for loading a previously generated ``.tfrecord`` dataset.

The main class is :py:class:`MergerNetDataset`, which is a high-level representation of the
dataset and can be used for all development process of this deep learning model.

This module defines others classes and functions as well, who perform complementary tasks.
"""


from dis import dis
from pathlib import Path
from posixpath import split
from typing import Dict, List

from mergernet.core.utils import load_image

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
import pandas as pd



# def parse_rgb_example(serialized_example):
#   """Parse rgb example."""
#   feature = {
#     'label': tf.io.FixedLenFeature([], tf.int64),
#     'image': tf.io.FixedLenFeature([], tf.string)
#   }
#   example = tf.io.parse_single_example(serialized_example, feature)
#   return tf.io.parse_tensor(example['image'], tf.int64), example['label']


def get_bytes_feature(value: str):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def get_int64_feature(value: int):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def get_float_feature(value: float):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def write_tfrecord_file(file_path: Path, examples: dict):
  with tf.io.TFRecordWriter(str(file_path.resolve())) as tfrecord:
    for i in range(len(examples['image'])):
      label = examples['label'][i]
      image_path = examples['image'][i]
      image_array = load_image(Path(image_path))
      image_tensor = tf.convert_to_tensor(image_array)
      image_serialized = tf.io.serialize_tensor(image_tensor).numpy() # serialized string

      features = {
        'label': get_int64_feature(label),
        'image': get_bytes_feature(image_serialized)
      }

      example = tf.train.Example(features=tf.train.Features(feature=features))
      tfrecord.write(example.SerializeToString())



def parser(serialized_example):
  features = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
  }

  example = tf.io.parse_single_example(
    serialized=serialized_example,
    features=features
  )

  image = example['image']
  label = example['label']

  return tf.io.parse_tensor(image, tf.int64), label



class DatasetConfig:
  """Configuration params for dataset."""
  def __init__(
    self,
    download_url: str = None,
    save_path: Path = None,
    train_glob: Path = None,
    validation_glob: Path = None,
    test_glob: Path = None
  ):
    self.download_url = download_url
    self.save_path = save_path
    self.train_glob = save_path.parent / train_glob # concatenate folder and glob
    self.validation_glob = save_path.parent / validation_glob
    self.test_glob = save_path.parent / test_glob



class DatasetCreator:
  def __init__(
    self,
    # images_folder: Path,
    images: List[Path],
    labels: List[int],
    # rmag: List[float],
    # metadata_table: Path,
    output: Path,
    n_splits: int = 5,
    bins: int = 6,
    # rmag_column: str = 'r',
    # label_column: str = 'class',
    # filename_column: str = 'filename'
  ):
    self.images = images
    self.labels = labels
    # self.rmag = rmag
    # self.images_folder = images_folder
    # self.metadata_table = metadata_table
    self.output = output
    self.n_splits = n_splits
    self.bins = bins
    # self.rmag_column = rmag_column
    # self.label_column = label_column
    # self.filename_column = filename_column


  def generate(self):
    kf = StratifiedKFold(n_splits=self.n_splits)
    for fold, (train_ids, test_ids) in enumerate(kf.split(self.images, self.labels)):
      img_train = self.images[train_ids]
      img_test = self.images[test_ids]
      lbl_train = self.labels[train_ids]
      lbl_test = self.labels[test_ids]

      train_examples = {
        'image': img_train,
        'label': lbl_train
      }

      test_examples = {
        'image': img_test,
        'label': lbl_test
      }

      write_tfrecord_file(
        file_path=self.output / f'train_fold_{fold}.tfrecord',
        examples=train_examples
      )

      write_tfrecord_file(
        file_path=self.output / f'test_fold_{fold}.tfrecord',
        examples=test_examples
      )

    # for fold in range(self.n_splits):
    #   train_img_path = splited_dataset['X_train'][fold]
    #   train_labels = splited_dataset['y_train'][fold]
    #   test_img_path = splited_dataset['X_test'][fold]
    #   test_labels = splited_dataset['y_test'][fold]

    #   train_examples = {
    #     'image': train_img_path,
    #     'labels': train_labels
    #   }

    #   test_examples = {
    #     'image': test_img_path,
    #     'labels': test_labels
    #   }

    #   write_tfrecord_file(
    #     file_path=self.output / f'train_fold_{fold}.tfrecord',
    #     examples=train_examples
    #   )

    #   write_tfrecord_file(
    #     file_path=self.output / f'test_fold_{fold}.tfrecord',
    #     examples=test_examples
    #   )


  def _generate(self):
    table = pd.read_csv(self.metadata_table)
    rmag = table[[self.rmag_column]].to_numpy()
    filenames = table[[self.id_column]].to_numpy()
    labels = table[[self.label_column]].to_numpy()

    kf = StratifiedDistributionKFold(n_splits=self.n_splits, bins=self.bins)
    splited_dataset = kf.split(filenames, labels)

    for fold in range(self.n_splits):
      train_filenames = splited_dataset['X_train'][fold]
      train_labels = splited_dataset['y_train'][fold]
      test_filenames = splited_dataset['X_test'][fold]
      test_labels = splited_dataset['y_test'][fold]

      train_examples = {
        'image': [self.images_folder / filename for filename in train_filenames],
        'labels': train_labels
      }

      test_examples = {
        'image': [self.image_folder / filename for filename in test_filenames],
        'labels': test_labels
      }

      write_tfrecord_file(
        file_path=self.output / f'train_fold_{fold}.tfrecord',
        examples=train_examples
      )

      write_tfrecord_file(
        file_path=self.output / f'test_fold_{fold}.tfrecord',
        examples=test_examples
      )



class DistributionKFold:
  def __init__(self, bins: int):
    self.bins = bins


  def split(self, distribution):
    bin_edges = np.histogram_bin_edges(distribution, bins=self.bins)

    for i in range(len(bin_edges) - 1):
      bin_ids = np.asarray(
        (distribution >= bin_edges[i]) & (distribution <= bin_edges[i + 1])
      ).nonzero()[0]
      yield bin_ids




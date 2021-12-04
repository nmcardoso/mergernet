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




from tarfile import XGLTYPE
from typing import Path
import os

import tensorflow as tf



def one_hot(X, y):
  return tf.one_hot(y)



def load_jpeg(X, *kargs):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_jpeg(img_bytes, channels=3)
  return img



def normalize_rgb(X, *kargs):
  return X / 255.




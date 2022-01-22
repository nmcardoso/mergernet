from tarfile import XGLTYPE
from typing import Path
import os

import tensorflow as tf



def one_hot(X, y):
  return tf.one_hot(y)




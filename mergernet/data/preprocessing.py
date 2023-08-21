import tensorflow as tf


def one_hot_factory(n_class):
  def one_hot(X, y):
    return X, tf.one_hot(y, n_class)
  return one_hot



def load_jpg_with_label(X, y=None):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_jpeg(img_bytes, channels=3)
  return img, y



def load_png_with_label(X, y=None):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_png(img_bytes, channels=3)
  return img, y



def load_jpg(X):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_jpeg(img_bytes, channels=3)
  return img



def load_png(X):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_png(img_bytes, channels=3)
  return img



def normalize_rgb(X, y=None):
  return X / 255., y



def standardize_rgb(X, y=None):
  return tf.image.per_image_standardization(X), y

import tensorflow as tf


def one_hot_factory(n_class):
  def one_hot(X, y):
    return X, tf.one_hot(y, n_class)
  return one_hot



def load_jpg(X, y):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_jpeg(img_bytes, channels=3)
  return img, y



def load_png(X, y):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_png(img_bytes, channels=3)
  return img, y



def normalize_rgb(X, y):
  return X / 255., y



def standardize_rgb(X, y):
  return tf.image.per_image_standardization(X), y

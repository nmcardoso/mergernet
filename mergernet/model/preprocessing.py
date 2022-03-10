import tensorflow as tf



def one_hot(X, y):
  depth = tf.math.reduce_max(y)
  return X, tf.one_hot(y, depth)



def load_jpg(X, y):
  img_bytes = tf.io.read_file(X)
  img = tf.io.decode_jpeg(img_bytes, channels=3)
  return img, y



def normalize_rgb(X, y):
  return X / 255., y



def standardize_rgb(X, y):
  return tf.image.per_image_standardization(X), y

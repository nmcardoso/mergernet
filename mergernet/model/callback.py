import tensorflow as tf



class DeltaStopping(tf.keras.callbacks.Callback):
  def __init__(self):
    super(DeltaStopping, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    if (epoch > 2) and (logs['accuracy'] - logs['val_accuracy']) > 0.1:
      self.model.stop_training = True

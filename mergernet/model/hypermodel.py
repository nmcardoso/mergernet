import logging
from typing import Tuple

import tensorflow as tf
import keras_tuner as kt

from mergernet.core.dataset import Dataset
from mergernet.core.utils import Timming
from mergernet.core.artifacts import ArtifactHelper
from mergernet.model.preprocessing import load_jpg, one_hot
from mergernet.model.callback import DeltaStopping


L = logging.getLogger('job')


class SimpleHyperModel():
  def __init__(self, dataset: Dataset):
    self.dataset = dataset
    self.ds_train: tf.data.Dataset = None
    self.ds_test: tf.data.Dataset = None
    self.class_weights: dict = None
    self._prepare_data()


  def build(
    self,
    input_shape: Tuple = (128, 128, 3),
    pretrainded_weights: str = 'imagenet',
    dense_units_1: int = 256,
    dropout_rate_1: float = 0.4,
    dense_2: bool = False,
    dense_units_2: int = None,
    # dropout_2: bool = False,
    dropout_rate_2: float = None,
    dense_3: bool = False,
    dense_units_3: int = None,
    # dropout_3: bool = False,
    dropout_rate_3: float = None,
    learning_rate: float = 5e-5
  ) -> tf.keras.Model:
    conv_block = tf.keras.applications.ResNet50(
      input_shape=input_shape,
      include_top=False,
      weights=pretrainded_weights
    )

    preprocess_input = tf.keras.applications.resnet.preprocess_input

    data_aug_layers = [
      tf.keras.layers.RandomFlip(mode='horizontal', seed=42),
      tf.keras.layers.RandomFlip(mode='vertical', seed=42),
      tf.keras.layers.RandomRotation(
        (-0.08, 0.08),
        fill_mode='reflect',
        interpolation='bilinear',
        seed=42
      ),
      tf.keras.layers.RandomZoom(
        (-0.15, 0.0),
        fill_mode='reflect',
        interpolation='bilinear',
        seed=42
      )
    ]

    data_aug_block = tf.keras.Sequential(data_aug_layers, name='data_augmentation')

    inputs = tf.keras.Input(shape=input_shape)
    x = data_aug_block(inputs)
    x = preprocess_input(x)
    x = conv_block(x)#, Training=False)
    # if dense_block is None:
    #   x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #   x = tf.keras.layers.Dropout(0.4)(x)
    # else:
    x = tf.keras.layers.Flatten()(x)
    # x = dense_block(x)
    x = tf.keras.layers.Dense(dense_units_1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate_1)(x)
    if dense_2:
      x = tf.keras.layers.Dense(dense_units_2, activation='relu')(x)
      if dropout_rate_2 > 0:
        x = tf.keras.layers.Dropout(dropout_rate_2)(x)

    if dense_3:
      x = tf.keras.layers.Dense(dense_units_3, activation='relu')(x)
      if dropout_rate_3 > 0:
        x = tf.keras.layers.Dropout(dropout_rate_3)(x)

    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
      optimizer=tf.keras.optimizers.Adam(
        learning_rate
      ),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
      ]
    )
    return model


  def fit(
    self,
    # model build params
    input_shape: Tuple = (128, 128, 3),
    pretrainded_weights: str = 'imagenet',
    dense_units_1: int = 256,
    dropout_rate_1: float = 0.4,
    dense_2: bool = False,
    dense_units_2: int = None,
    # dropout_2: bool = False,
    dropout_rate_2: float = None,
    dense_3: bool = False,
    dense_units_3: int = None,
    # dropout_3: bool = False,
    dropout_rate_3: float = None,
    learning_rate: float = 5e-5,
    # train params
    batch_size: int = 64,
    epochs: int = 12
  ):
    if self.ds_train is None:
      self._prepare_data()

    model = self.build(
      input_shape=input_shape,
      pretrainded_weights=pretrainded_weights,
      dense_units_1=dense_units_1,
      dropout_rate_1=dropout_rate_1,
      dense_2=dense_2,
      dense_units_2=dense_units_2,
      # dropout_2=dropout_2,
      dropout_rate_2=dropout_rate_2,
      dense_3=dense_3,
      dense_units_3=dense_units_3,
      # dropout_3=dropout_3,
      dropout_rate_3=dropout_rate_3,
      learning_rate=learning_rate
    )

    t = Timming()
    t.start()
    L.info('[TRAIN] Starting training loop')

    ah = ArtifactHelper()

    history = model.fit(
      self.ds_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=self.ds_test,
      class_weight=self.class_weights,
      callbacks=[
        tf.keras.callbacks.TensorBoard(
          log_dir=str(ah.artifact_path / 'tensorboard'),
          write_images=True,
          profile_batch=40
        ),
        tf.keras.callbacks.EarlyStopping(patience=3),
        DeltaStopping()
      ]
    )

    t.end()
    L.info(f'[TRAIN] Training finished without errors in {t.duration()}.')

    ev = model.evaluate(self.ds_test)

    return ev



  def _prepare_data(self):
    ds_train, ds_test = self.dataset.get_fold(0)
    _x, _y = next(ds_train.take(1).as_numpy_iterator())
    L.info('[DATASET] Fold 0 loaded')

    ds_train = ds_train.map(load_jpg)
    ds_test = ds_test.map(load_jpg)
    _x, _y = next(ds_train.take(1).as_numpy_iterator())
    L.debug('[DATASET] apply: load_jpg')
    L.debug(f'[DATASET] Example shape (X, y): {_x.shape}, {_y.shape}')

    ds_train = ds_train.map(one_hot)
    ds_test = ds_test.map(one_hot)
    _x, _y = next(ds_train.take(1).as_numpy_iterator())
    L.debug('[DATASET] apply: one_hot')
    L.debug(f'[DATASET] Example shape (X, y): {_x.shape}, {_y.shape}')

    # ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(5000)
    ds_test = ds_test.shuffle(1000)
    _x, _y = next(ds_train.take(1).as_numpy_iterator())
    L.debug('[DATASET] apply: shuffle')
    L.debug(f'[DATASET] Example shape (X, y): {_x.shape}, {_y.shape}')

    ds_train = ds_train.batch(64)
    ds_test = ds_test.batch(64)
    _x, _y = next(ds_train.take(1).as_numpy_iterator())
    L.debug('[DATASET] apply: batch')
    L.debug(f'[DATASET] Example shape (X, y): {_x.shape}, {_y.shape}')

    _x, _y = next(ds_test.take(1).as_numpy_iterator())
    L.debug('[DATASET] apply: batch')
    L.debug(f'[DATASET] Test example shape (X, y): {_x.shape}, {_y.shape}')

    self.ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    self.ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    self.class_weights = self.dataset.compute_class_weight()


# dense_units_1: hp.Choice('dense_1_units', [256, 512, 1024])
    # dropout_rate_1: hp.Choice('dropout_1_rate', [0.2, 0.3, 0.4, 0.5])
    # dense_2: hp.Boolean('dense_2')
    # dense_units_2: hp.Choice('dense_2_units', [128, 256, 512])
    # dropout_2: hp.Boolean('dropout_2')
    # dropout_rate_2: hp.Choice('dropout_2_rate', [0.2, 0.3, 0.4, 0.5])
    # dense_3: hp.Boolean('dense_3')
    # dense_units_3: hp.Choice('dense_3_units', [64, 128, 256, 512])
    # dropout_3: hp.Boolean('dropout_3')
    # dropout_rate_3: hp.Choice('dropout_3_rate', [0.2, 0.3, 0.4, 0.5])


class BayesianTuner(kt.BayesianOptimization):
  def run_trial(self, trial, *kargs, **kwargs):
    dataset = kwargs['dataset']

    hp = trial.hyperparameters
    sm = SimpleHyperModel(dataset)

    return sm.fit(
      dense_units_1=hp.Choice('dense_1_units', [256, 512, 1024]),
      dropout_rate_1=hp.Choice('dropout_1_rate', [0.2, 0.3, 0.4, 0.5]),
      dense_2=hp.Boolean('dense_2'),
      dense_units_2=hp.Choice('dense_2_units', [128, 256, 512], parent_name='dense_2', parent_values=[True]),
      # dropout_2=hp.Boolean('dropout_2', parent_name='dense_2', parent_values=[True]),
      dropout_rate_2=hp.Choice('dropout_2_rate', [-1.0, 0.2, 0.3, 0.4, 0.5], parent_name='dense_2', parent_values=[True]),
      dense_3=hp.Boolean('dense_3', parent_name='dense_2', parent_values=[True]),
      dense_units_3=hp.Choice('dense_3_units', [64, 128, 256, 512], parent_name='dense_3', parent_values=[True]),
      # dropout_3=hp.Boolean('dropout_3', parent_name='dense_3', parent_values=[True]),
      dropout_rate_3=hp.Choice('dropout_3_rate', [-1.0, 0.2, 0.3, 0.4, 0.5], parent_name='dense_3', parent_values=[True]),
      learning_rate=hp.Choice('learning_rate', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
      epochs=1
    )

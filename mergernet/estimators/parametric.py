import logging
from typing import List, Tuple

import tensorflow as tf
from wandb.keras import WandbMetricsLogger

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.estimators.base import Estimator
from mergernet.model.callbacks import WandbGraphicsCallback

L = logging.getLogger(__name__)


class ParametricEstimator(Estimator):
  def __init__(self, hp: HyperParameterSet, dataset: Dataset):
    super().__init__(hp, dataset)


  def build(self, freeze_conv: bool = False) -> tf.keras.Model:
    conv_arch, preprocess_input = self.get_conv_arch(
      self.hp.get('architecture')
    )
    conv_block = conv_arch(
      input_shape=self.dataset.config.image_shape,
      include_top=False,
      weights=self.hp.get('pretrained_weights'),
    )
    conv_block._name = 'conv_block'
    conv_block.trainable = (not freeze_conv)
    L.info(f'Trainable weights of convolutional block: {len(conv_block.trainable_weights)}')

    data_aug_block = self.get_dataaug_block(
      flip_horizontal=True,
      flip_vertical=True,
      rotation=(-0.08, 0.08),
      zoom=False
    )

    inputs = tf.keras.Input(shape=self.dataset.config.image_shape)
    x = data_aug_block(inputs)
    x = preprocess_input(x)
    x = conv_block(x)
    x = tf.keras.layers.Flatten()(x)
    if self.hp.get('dense_1_units'):
      x = tf.keras.layers.Dense(self.hp.get('dense_1_units'), activation='relu')(x)
    if self.hp.get('dropout_1_rate'):
      x = tf.keras.layers.Dropout(self.hp.get('dropout_1_rate'))(x)
    if self.hp.get('dense_2_units'):
      x = tf.keras.layers.Dense(self.hp.get('dense_2_units'), activation='relu')(x)
    if self.hp.get('dropout_2_rate'):
      x = tf.keras.layers.Dropout(self.hp.get('dropout_2_rate'))(x)
    if self.hp.get('dense_3_units'):
      x = tf.keras.layers.Dense(self.hp.get('dense_3_units'), activation='relu')(x)
    if self.hp.get('dropout_3_rate'):
      x = tf.keras.layers.Dropout(self.hp.get('dropout_3_rate'))(x)
    outputs = tf.keras.layers.Dense(self.dataset.config.n_classes)(x)

    self._tf_model = tf.keras.Model(inputs, outputs)
    L.info(f'Trainable weights (TOTAL): {len(self._tf_model.trainable_weights)}')

    return self._tf_model


  def train(
    self,
    run_name: str = 'run-0',
    callbacks: List[tf.keras.callbacks.Callback] = [],
  ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    tf.keras.backend.clear_session()

    ds_train, ds_test = self.dataset.get_fold(0)
    ds_train = self.dataset.prepare_data(
      ds_train,
      batch_size=self.hp.get('batch_size'),
      buffer_size=5000,
      kind='train'
    )
    ds_test = self.dataset.prepare_data(
      ds_test,
      batch_size=self.hp.get('batch_size'),
      buffer_size=1000,
      kind='train'
    )

    class_weights = self.dataset.compute_class_weight()

    model = self.build(freeze_conv=True)

    opt = self.get_optimizer(self.hp.get('t1_opt'), lr=self.hp.get('t1_lr'))
    self.compile_model(
      model,
      optimizer=opt,
      label_smoothing=self.hp.get('label_smoothing', default=0.0),
    )

    with Experiment.Tracker(self.hp.to_values_dict(), name=run_name, job_type='train'):
      early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        mode='min', # 'min' or 'max'
        restore_best_weights=True
      )

      # wandb_cb = MyWandbCallback(
      #   validation_data=ds_test,
      #   labels=self.dataset.config.labels,
      #   monitor='val_loss',
      #   mode='min',
      # )

      wandb_metrics = WandbMetricsLogger()
      wandb_graphics = WandbGraphicsCallback(
        validation_data=ds_test,
        labels=self.dataset.config.labels
      )

      t1_epochs = self.hp.get('tl_epochs', default=10)
      batch_size = self.hp.get('batch_size')

      t = Timming()
      L.info('Start of training loop with frozen CNN')
      h = model.fit(
        ds_train,
        batch_size=batch_size,
        epochs=t1_epochs,
        validation_data=ds_test,
        class_weight=class_weights,
        callbacks=[early_stop_cb, wandb_metrics, wandb_graphics]
      )
      L.info(f'End of training loop, duration: {t.end()}')
      L.info(f'History keys: {", ".join(h.history.keys())}')
      L.info(f'History length: {len(h.history["loss"])}')

      self.set_trainable(model, 'conv_block', True)

      lr_scheduler = self.get_scheduler(
        self.hp.get('lr_decay'),
        lr=self.hp.get('opt_lr')
      )
      lr = lr_scheduler or self.hp.get('opt_lr')
      opt = self.get_optimizer(self.hp.get('optimizer'), lr=lr)

      self.compile_model(
        model,
        optimizer=opt,
        label_smoothing=self.hp.get('label_smoothing', default=0.0),
      )

      t = Timming()
      L.info('Start of main training loop')
      model.fit(
        ds_train,
        batch_size=batch_size,
        epochs=len(h.history['loss']) + self.hp.get('epochs'),
        validation_data=ds_test,
        class_weight=class_weights,
        initial_epoch=len(h.history['loss']),
        callbacks=[wandb_metrics, wandb_graphics, *callbacks],
      )
      L.info(f'End of training loop, duration: {t.end()}')

      self._tf_model = model
    return self._tf_model


  def predict(self):
    pass

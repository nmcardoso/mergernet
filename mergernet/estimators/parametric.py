import logging
from typing import List, Tuple

import tensorflow as tf
import wandb

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.estimators.base import Estimator
from mergernet.model.callbacks import MyWandbCallback

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
    x = tf.keras.layers.Dense(self.hp.get('dense_1_units'), activation='relu')(x)
    x = tf.keras.layers.Dropout(self.hp.get('dropout_1_rate'))(x)
    x = tf.keras.layers.Dense(self.hp.get('dense_2_units'), activation='relu')(x)
    x = tf.keras.layers.Dropout(self.hp.get('dropout_2_rate'))(x)
    outputs = tf.keras.layers.Dense(self.dataset.config.n_classes, activation='softmax')(x)

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
    self.compile_model(model, tf.keras.optimizers.Adam(self.hp.get('opt_lr')))

    with Experiment.Tracer(self.hp.to_values_dict(), name=run_name, job_type='train'):
      early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        mode='min', # 'min' or 'max'
        restore_best_weights=True
      )

      wandb_cb = MyWandbCallback(
        dataset=self.dataset,
        monitor='val_loss',
        mode='min',
        save_graph=True,
        save_model=False,
        log_weights=False,
        log_gradients=False,
        compute_flops=True,
      )

      t = Timming()
      L.info('Start of training loop with frozen CNN')
      h1 = model.fit(
        ds_train,
        batch_size=self.hp.get('batch_size'),
        epochs=self.hp.get('tl_epochs', default=10),
        validation_data=ds_test,
        class_weight=class_weights,
        callbacks=[early_stop_cb, wandb_cb, *callbacks]
      )
      L.info(f'End of training loop, duration: {t.end()}')

      self.set_trainable(model, 'conv_block', True)
      self.compile_model(model, tf.keras.optimizers.Adam(self.hp.get('opt_lr')))

      t = Timming()
      L.info('Start of main training loop')
      model.fit(
        ds_train,
        batch_size=self.hp.get('batch_size'),
        epochs=self.hp.get('tl_epochs', default=10) + self.hp.get('epochs'),
        validation_data=ds_test,
        class_weight=class_weights,
        initial_epoch=len(h1.history['loss']),
        callbacks=[wandb_cb, *self.callbacks],
      )
      L.info(f'End of training loop, duration: {t.end()}')

      self._tf_model = model
    return self._tf_model


  def predict(self):
    pass

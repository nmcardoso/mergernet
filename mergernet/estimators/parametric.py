import logging
from pathlib import Path
from typing import List, Tuple, Union

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
    L.info(f'Conv block weights: {len(conv_block.weights)}')
    L.info(f'Conv block trainable weights: {len(conv_block.trainable_weights)}')
    L.info(f'Conv block non-trainable weights: {len(conv_block.non_trainable_weights)}')

    data_aug_block = self.get_dataaug_block(
      flip_horizontal=True,
      flip_vertical=True,
      rotation=(-0.08, 0.08),
      zoom=False
    )

    # Input
    inputs = tf.keras.Input(shape=self.dataset.config.image_shape)

    # Data Augmentation
    x = data_aug_block(inputs)

    # Input pre-processing
    x = preprocess_input(x)

    # Feature extractor
    x = conv_block(x)

    # Representation layer
    representation_mode = self.hp.get('features_reduction', default='flatten')
    if representation_mode == 'flatten':
      x = tf.keras.layers.Flatten()(x)
    elif representation_mode == 'avg_pooling':
      x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif representation_mode == 'max_pooling':
      x = tf.keras.layers.GlobalMaxPooling2D()(x)
    if self.hp.get('batch_norm_0'):
      x = tf.keras.layers.BatchNormalization()(x)
    if self.hp.get('dropout_0_rate'):
      x = tf.keras.layers.Dropout(self.hp.get('dropout_0_rate'))(x)

    # Classifier
    for i in range(1, 4):
      units = self.hp.get(f'dense_{i}_units')
      bn = self.hp.get(f'batch_norm_{i}')
      activation = self.hp.get(f'activation_{i}', default='relu')
      dropout_rate = self.hp.get(f'dropout_{i}_rate')
      if units:
        x = tf.keras.layers.Dense(units, use_bias=not bn)(x)
        if bn:
          x = tf.keras.layers.BatchNormalization()(x)
        if activation == 'relu':
          x = tf.keras.layers.Activation('relu')(x)
        if dropout_rate:
          x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Classifications
    outputs = tf.keras.layers.Dense(self.dataset.config.n_classes, activation='softmax')(x)

    self._tf_model = tf.keras.Model(inputs, outputs)
    L.info(f'Final model weights: {len(self._tf_model.weights)}')
    L.info(f'Final model trainable weights: {len(self._tf_model.trainable_weights)}')
    L.info(f'Final model non-trainable weights: {len(self._tf_model.non_trainable_weights)}')

    return self._tf_model


  def train(
    self,
    run_name: str = 'run-0',
    callbacks: List[tf.keras.callbacks.Callback] = [],
    fold: int = 0,
  ) -> tf.keras.Model:
    tf.keras.backend.clear_session()

    with Experiment.Tracker(self.hp.to_values_dict(), name=run_name, job_type='train'):
      # dataset preparation
      ds_train, ds_test = self.dataset.get_fold(fold)
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
      self.ds_train = ds_train
      self.ds_test = ds_test

      class_weights = self.dataset.compute_class_weight()

      # w&b callbacks
      wandb_metrics = WandbMetricsLogger()
      wandb_graphics = WandbGraphicsCallback(
        validation_data=ds_test,
        labels=self.dataset.config.labels
      )

      # t1 train
      t1_epochs = 0
      if self.hp.get('t1_epochs', default=0) > 0:
        model = self.build(freeze_conv=True)

        opt = self.get_optimizer(self.hp.get('t1_opt'), lr=self.hp.get('t1_lr'))
        self.compile_model(
          model,
          optimizer=opt,
          label_smoothing=self.hp.get('label_smoothing', default=0.0),
        )

        early_stop_cb = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          min_delta=0,
          patience=2,
          mode='min', # 'min' or 'max'
          restore_best_weights=True
        )

        t1_epochs = self.hp.get('tl_epochs', default=10)

        t = Timming()
        L.info('Start of training loop with frozen CNN')
        h = model.fit(
          ds_train,
          batch_size=self.hp.get('batch_size'),
          epochs=t1_epochs,
          validation_data=ds_test,
          class_weight=class_weights,
          callbacks=[early_stop_cb, wandb_metrics, wandb_graphics]
        )
        L.info(f'End of training loop, duration: {t.end()}')
        L.info(f'History keys: {", ".join(h.history.keys())}')
        L.info(f'History length: {len(h.history["loss"])}')

        self.set_trainable(model, 'conv_block', True)
        t1_epochs += len(h.history['loss'])
      else:
        model = self.build(freeze_conv=False)

      # main train
      lr_scheduler = self.get_scheduler(
        self.hp.get('lr_decay'),
        lr=self.hp.get('opt_lr')
      )
      lr = lr_scheduler or self.hp.get('opt_lr')
      L.info(f'Using learning rate: {str(lr_scheduler)}')
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
        batch_size=self.hp.get('batch_size'),
        epochs=t1_epochs + self.hp.get('epochs'),
        validation_data=ds_test,
        class_weight=class_weights,
        initial_epoch=t1_epochs,
        callbacks=[wandb_metrics, wandb_graphics, *callbacks],
      )
      L.info(f'End of training loop, duration: {t.end()}')

      self._tf_model = model
    return self._tf_model


  def predict(self):
    pass


  def cross_validation(
    self,
    run_name: str = 'run-0',
    callbacks: List[tf.keras.callbacks.Callback] = []
  ):
    for fold in range(self.dataset.get_n_folds()):
      model = self.train(run_name=f'{run_name}_fold-{fold}', callbacks=callbacks)
      preds = model.predict(self.ds_test)

      # for label, index in label_map.items():
      # y_hat = [pred[index] for pred in self._preds]
      # print('y_hat_len', len(y_hat))
      # df[f'prob_{label}'] = y_hat

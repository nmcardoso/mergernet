import logging
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import wandb

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.model.utils import (get_conv_arch, set_trainable_state,
                                   setup_seeds)

setup_seeds()

L = logging.getLogger(__name__)




def finetune_train(
  dataset: Dataset,
  hp: HyperParameterSet,
  callbacks: List[tf.keras.callbacks.Callback] = None,
  run_name: str = 'run'
) -> tf.keras.Model:
  tf.keras.backend.clear_session()

  ds_train, ds_test = dataset.get_fold(0)
  ds_train = dataset.prepare_data(
    ds_train,
    batch_size=hp.get('batch_size'),
    buffer_size=5000,
    kind='train'
  )
  ds_test = dataset.prepare_data(
    ds_test,
    batch_size=hp.get('batch_size'),
    buffer_size=1000,
    kind='train'
  )

  class_weights = dataset.compute_class_weight()

  model = _build_model(
    input_shape=dataset.config.image_shape,
    n_classes=dataset.config.n_classes,
    freeze_conv=True,
    hp=hp
  )
  _compile_model(model, tf.keras.optimizers.Adam(hp.get('opt_lr')))

  early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=2,
    mode='min', # 'min' or 'max'
    restore_best_weights=True
  )

  with Experiment.Tracer(hp.to_values_dict(), name=run_name, job_type='train'):
    wandb_cb = wandb.keras.WandbCallback(
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
      batch_size=hp.get('batch_size'),
      epochs=10,
      validation_data=ds_test,
      class_weight=class_weights,
      callbacks=[early_stop_cb, wandb_cb]
    )
    L.info(f'End of training loop, duration: {t.end()}')

    set_trainable_state(model, 'conv_block', True)
    _compile_model(model, tf.keras.optimizers.Adam(hp.get('opt_lr')))

    t = Timming()
    L.info('Start of main training loop')
    model.fit(
      ds_train,
      batch_size=hp.get('batch_size'),
      epochs=hp.get('epochs'),
      validation_data=ds_test,
      class_weight=class_weights,
      initial_epoch=len(h1.history),
      callbacks=[wandb_cb, *callbacks],
    )
    L.info(f'End of training loop, duration: {t.end()}')

  return model



def _build_model(
  input_shape: Tuple,
  n_classes: int,
  freeze_conv: bool = False,
  hp: HyperParameterSet = None
) -> tf.keras.Model:
  # dataset.config.n_classes
  conv_arch, preprocess_input = get_conv_arch(
    hp.get('architecture')
  )
  conv_block = conv_arch(
    input_shape=input_shape,
    include_top=False,
    weights=hp.get('pretrained_weights'),
  )
  conv_block._name = 'conv_block'
  conv_block.trainable = (not freeze_conv)
  L.info(f'Trainable weights (CONV): {len(conv_block.trainable_weights)}')

  data_aug_layers = [
    tf.keras.layers.RandomFlip(mode='horizontal', seed=RANDOM_SEED),
    tf.keras.layers.RandomFlip(mode='vertical', seed=RANDOM_SEED),
    tf.keras.layers.RandomRotation(
      (-0.08, 0.08),
      fill_mode='reflect',
      interpolation='bilinear',
      seed=RANDOM_SEED
    ),
    tf.keras.layers.RandomZoom(
      (-0.15, 0.0),
      fill_mode='reflect',
      interpolation='bilinear',
      seed=RANDOM_SEED
    )
  ]
  data_aug_block = tf.keras.Sequential(data_aug_layers, name='data_augmentation')

  inputs = tf.keras.Input(shape=input_shape)
  x = data_aug_block(inputs)
  x = preprocess_input(x)
  x = conv_block(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(hp.get('dense_1_units'), activation='relu')(x)
  x = tf.keras.layers.Dropout(hp.get('dropout_1_rate'))(x)
  x = tf.keras.layers.Dense(hp.get('dense_2_units'), activation='relu')(x)
  x = tf.keras.layers.Dropout(hp.get('dropout_2_rate'))(x)
  outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

  model = tf.keras.Model(inputs, outputs)
  L.info(f'Trainable weights (TOTAL): {len(model.trainable_weights)}')

  return model



def _compile_model(
  model: tf.keras.Model,
  optimizer: tf.keras.optimizers.Optimizer,
  metrics: list = []
):
  model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      *metrics
    ]
  )

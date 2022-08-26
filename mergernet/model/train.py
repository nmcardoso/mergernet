import logging
import random
import os
from typing import Any, Callable, Dict, Tuple, Union
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.dataset import Dataset
from mergernet.core.entity import ConstantHyperParameter, HyperParameterSet
from mergernet.core.experiment import Experiment
from mergernet.model.callback import PruneCallback, SaveCallback
from mergernet.model.preprocessing import load_jpg, load_png, one_hot_factory
from mergernet.core.utils import Timming
from mergernet.model.utils import get_conv_arch, set_trainable_state


L = logging.getLogger(__name__)


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)



def finetune_train(dataset: Dataset, hp: HyperParameterSet):
  tf.keras.backend.clear_session()
  e = Experiment()

  ds_train, ds_test = dataset.get_fold(0)
  ds_train = dataset.prepare_data(
    ds_train,
    batch_size=hp.batch_size.suggest(),
    buffer_size=5000,
    kind='train'
  )
  ds_test = dataset.prepare_data(
    ds_test,
    batch_size=hp.batch_size.suggest(),
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
  _compile_model(model, tf.keras.optimizers.Adam(hp.opt_lr.suggest()))

  ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    Path(e.local_artifact_path) / f'model.ckpt.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min' # 'min' or 'max'
  )

  early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=2,
    mode='min', # 'min' or 'max'
    restore_best_weights=True
  )

  t = Timming()
  L.info('Start of training loop')
  h1 = model.fit(
    ds_train,
    batch_size=hp.batch_size.suggest(),
    epochs=10,
    validation_data=ds_test,
    class_weight=class_weights,
    callbacks=[early_stop_cb]
  )
  L.info(f'End of training loop, duration: {t.end()}.')

  set_trainable_state(model, 'conv_block', True)
  _compile_model(model, tf.keras.optimizers.Adam(hp.opt_lr.suggest()))

  t = Timming()
  L.info('Start of training loop')
  model.fit(
    ds_train,
    batch_size=hp.batch_size.suggest(),
    epochs=hp.epochs.suggest(),
    validation_data=ds_test,
    class_weight=class_weights,
    initial_epoch=len(h1.history),
  )
  L.info(f'End of training loop, duration: {t.end()}.')

  return model



def _build_model(
  input_shape: Tuple,
  n_classes: int,
  freeze_conv: bool = False,
  hp: HyperParameterSet = None
) -> tf.keras.Model:
  # dataset.config.n_classes
  conv_arch, preprocess_input = get_conv_arch(
    hp.architecture.suggest()
  )
  conv_block = conv_arch(
    input_shape=input_shape,
    include_top=False,
    weights=hp.pretrained_weights.suggest(),
  )
  conv_block._name = 'conv_block'
  print(conv_block.name)
  print(conv_block.name)
  print(conv_block.name)
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
  x = tf.keras.layers.Dense(hp.dense_1_units.suggest(), activation='relu')(x)
  x = tf.keras.layers.Dropout(hp.dropout_1_rate.suggest())(x)
  x = tf.keras.layers.Dense(hp.dense_2_units.suggest(), activation='relu')(x)
  x = tf.keras.layers.Dropout(hp.dropout_2_rate.suggest())(x)
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

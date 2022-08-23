import logging
from typing import Any, Callable, Dict, Tuple, Union
from pathlib import Path

import optuna
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


L = logging.getLogger(__name__)
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()



class HyperModel:
  def __init__(
    self,
    dataset: Dataset,
    name: str,
    epochs: int = 20,
    nest_trials: bool = False
  ):
    self.dataset = dataset
    self.name = name
    self.epochs = epochs
    self.nest_trials = nest_trials
    self.hp: HyperParameterSet = None
    self.study: optuna.study.Study = None
    self.save_model: bool = None
    self.objective_metric: str = 'val_loss'
    self.objective_direction: str = 'minimize'
    self.class_weights: dict = None


  def prepare_data(
    self,
    ds: tf.data.Dataset,
    batch_size: int = 64,
    buffer_size: int = 1000,
    kind='train'
  ):
    if self.dataset.config.X_column_suffix == '.jpg':
      ds = ds.map(load_jpg)
      L.info('Apply: load_jpg')
    elif self.dataset.config.X_column_suffix == '.png':
      ds = ds.map(load_png)
      L.info('Apply: load_png')

    if kind == 'train':
      ds = ds.map(one_hot_factory(self.dataset.config.n_classes))
      L.info('Apply: one_hot')

    ds = ds.cache()
    L.info('Apply: cache')

    ds = ds.shuffle(buffer_size)
    L.info('Apply: shuffle')

    ds = ds.batch(batch_size)
    L.info('Apply: batch')

    ds = ds.prefetch(tf.data.AUTOTUNE)
    L.info('Apply: prefetch')

    return ds


  def build_model(
    self,
    input_shape: Tuple,
    trial: optuna.trial.FrozenTrial = None,
    freeze_conv: bool = False
  ) -> tf.keras.Model:
    if trial is not None:
      self.hp.set_trial(trial)

    conv_arch, preprocess_input = self._architecture_switch(
      self.hp.architecture.suggest()
    )
    conv_block = conv_arch(
      input_shape=input_shape,
      include_top=False,
      weights=self.hp.pretrained_weights.suggest()
    )
    # conv_block.name = 'conv_block'
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
    x = tf.keras.layers.Dense(self.hp.dense_1_units.suggest(), activation='relu')(x)
    x = tf.keras.layers.Dropout(self.hp.dropout_1_rate.suggest())(x)
    x = tf.keras.layers.Dense(self.hp.dense_2_units.suggest(), activation='relu')(x)
    x = tf.keras.layers.Dropout(self.hp.dropout_2_rate.suggest())(x)
    outputs = tf.keras.layers.Dense(self.dataset.config.n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    L.info(f'Trainable weights (TOTAL): {len(model.trainable_weights)}')
    model.compile(
      optimizer=tf.keras.optimizers.Adam(self.hp.learning_rate.suggest()),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
      ]
    )
    return model


  def _compile_model(self, model: tf.keras.Model, lr: None):
    lr = lr or self.hp.learning_rate.suggest()
    model.compile(
      optimizer=tf.keras.optimizers.Adam(lr),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
      ]
    )


  def _switch_trainable_status(
    self,
    model: tf.keras.Model,
    layer: str,
    trainable: bool
  ):
    for l in model.layers:
      if l.name == layer:
        l.trainable = trainable


  def predict(
    self,
    model: Union[str, tf.keras.Model],
    dataset: tf.data.Dataset
  ) -> list:
    tf.keras.backend.clear_session()

    if isinstance(model, str):
      model = tf.keras.models.load_model(Path(Experiment().local_artifact_path) / model)

    ds = dataset.get_preds_dataset()
    ds = self.prepare_data(ds, batch_size=128, buffer_size=1000, kind='predict')

    preds = model.predict(ds)

    return preds


  def _objective_old(self, trial: optuna.trial.FrozenTrial) -> float:
    tf.keras.backend.clear_session()

    ds_train, ds_test = self.dataset.get_fold(0)
    L.info('Preparing train dataset')
    ds_train = self.prepare_data(
      ds_train,
      batch_size=self.hp.batch_size.suggest(trial),
      buffer_size=5000,
      kind='train'
    )
    L.info('Preparing test dataset')
    ds_test = self.prepare_data(
      ds_test,
      batch_size=self.hp.batch_size.suggest(trial),
      buffer_size=1000,
      kind='train'
    )

    self._compute_class_weight()

    model = self.build_model(
      input_shape=self.dataset.config.image_shape,
      trial=trial
    )

    callbacks = [
      PruneCallback(trial=trial, objective_metric=self.objective_metric)
    ]

    if self.save_model:
      callbacks.append(
        SaveCallback(
          name=self.name,
          study=self.study,
          objective_metric=self.objective_metric,
          objective_direction=self.objective_direction
        )
      )

    t = Timming()
    L.info('Start of training loop')
    history = model.fit(
      ds_train,
      batch_size=self.hp.batch_size.suggest(trial),
      epochs=self.epochs,
      validation_data=ds_test,
      callbacks=callbacks
    )
    L.info(f'End of training loop, duration: {t.end()}.')

    # history shape: {metric1: [val1, val2, ...], metric2: [val1, val2, ...]}
    h = history.history
    # epochs = len(h[list(h.keys())[0]])

    # generating optuna value to optimize (val_accuracy)
    last_epoch_accuracy = h[self.objective_metric][-1]
    return last_epoch_accuracy



  def objective(self, trial: optuna.trial.FrozenTrial) -> float:
    tf.keras.backend.clear_session()
    e = Experiment()

    ds_train, ds_test = self.dataset.get_fold(0)
    ds_train = self.prepare_data(
      ds_train,
      batch_size=self.hp.batch_size.suggest(trial),
      buffer_size=5000,
      kind='train'
    )
    ds_test = self.prepare_data(
      ds_test,
      batch_size=self.hp.batch_size.suggest(trial),
      buffer_size=1000,
      kind='train'
    )
    self._compute_class_weight()

    model = self.build_model(
      input_shape=self.dataset.config.image_shape,
      trial=trial,
      freeze_conv=True
    )
    self._compile_model(model, self.hp.learning_rate.suggest() / 10)

    ckpt_path = Path(e.local_artifact_path) / f'{self.name}.ckpt.h5'

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path,
      monitor=self.objective_metric,
      save_best_only=True,
      mode=self.objective_direction[:3] # 'min' or 'max'
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
      monitor=self.objective_metric,
      min_delta=0,
      patience=2,
      mode=self.objective_direction[:3], # 'min' or 'max'
      restore_best_weights=True
    )

    prune_cb = PruneCallback(
      trial=trial,
      objective_metric=self.objective_metric
    )

    h1 = self._train(
      model,
      ds_train,
      ds_test,
      trial,
      callbacks=[early_stop_cb],
      save_model=False,
      epochs=10
    )

    self._switch_trainable_status(model, 'conv_block', True)
    self._compile_model(model, self.hp.learning_rate.suggest())

    h = self._train(
      model,
      ds_train,
      ds_test,
      trial,
      callbacks=[prune_cb],
      save_model=True,
      initial_epoch=len(h1),
      epochs=self.epochs
    )

    # generating optuna value to optimize (val_accuracy)
    objective_value = h[self.objective_metric][-1]
    return objective_value


  def _train(
    self,
    model: tf.keras.Model,
    ds_train: Dataset,
    ds_test: Dataset,
    trial: optuna.trial.FrozenTrial,
    save_model: bool,
    callbacks: list = [],
    initial_epoch: int = 0,
    epochs: int = 10,
  ):
    if save_model:
      callbacks.append(
        SaveCallback(
          name=self.name,
          study=self.study,
          objective_metric=self.objective_metric,
          objective_direction=self.objective_direction
        )
      )

    t = Timming()
    L.info('Start of training loop')
    history = model.fit(
      ds_train,
      batch_size=self.hp.batch_size.suggest(trial),
      epochs=epochs,
      validation_data=ds_test,
      class_weight=self.class_weights,
      callbacks=callbacks,
      initial_epoch=initial_epoch
    )
    L.info(f'End of training loop, duration: {t.end()}.')

    # history shape: {metric1: [val1, val2, ...], metric2: [val1, val2, ...]}
    h = history.history
    return h


  def hypertrain(
    self,
    n_trials: int,
    hyperparameters: HyperParameterSet,
    pruner: str = 'hyperband',
    objective_metric: str = 'val_loss',
    objective_direction: str = 'minimize',
    resume: bool = False,
    save_model: bool = True
  ):
    self.hp = hyperparameters
    self.save_model = save_model
    self.objective_metric = objective_metric
    self.objective_direction = objective_direction

    if resume:
      L.info(f'resuming previous optimization of {self.name} study')

    if pruner == 'median':
      pruner_instance = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10
      )
    elif pruner == 'hyperband':
      pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

    L.info(f'start of optuna optimization')

    exp = Experiment()
    optuna_path = Path(exp.local_artifact_path) / 'optuna.sqlite'
    optuna_uri = f'sqlite:///{str(optuna_path.resolve())}' # absolute path

    t = Timming()
    study = optuna.create_study(
      storage=optuna_uri,
      study_name=self.name,
      pruner=pruner_instance,
      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
      direction=objective_direction,
      load_if_exists=resume
    )
    self.study = study

    optimize_params = {
      'func': self.objective,
      'n_trials': n_trials
    }

    study.optimize(**optimize_params)
    t.end()

    exp.upload_file_gh('optuna.sqlite')

    L.info(f'optuna optimization finished in {t.duration()}')
    L.info(f'number of finished trials: {len(study.trials)}')
    L.info(f'----- begin of best trial summary -----')
    L.info(f'optimization score: {study.best_trial.value}')
    L.info(f'params:')
    for k, v in study.best_params.items():
      L.info(f'{k}: {str(v)}')
    L.info(f'----- end of best trial summary -----')


  def _compute_class_weight(self):
    if self.class_weights is None:
      self.class_weights = self.dataset.compute_class_weight()


  def _architecture_switch(self, pretrained_arch: str) -> Tuple[Callable, Callable]:
    if pretrained_arch == 'xception':
      preprocess_input = tf.keras.applications.xception.preprocess_input
      base_model = tf.keras.applications.Xception
    elif pretrained_arch == 'vgg16':
      preprocess_input = tf.keras.applications.vgg16.preprocess_input
      base_model = tf.keras.applications.VGG16
    elif pretrained_arch == 'vgg19':
      preprocess_input = tf.keras.applications.vgg19.preprocess_input
      base_model = tf.keras.applications.VGG19
    elif pretrained_arch == 'resnet50':
      preprocess_input = tf.keras.applications.resnet.preprocess_input
      base_model = tf.keras.applications.ResNet50
    elif pretrained_arch == 'resnet101':
      preprocess_input = tf.keras.applications.resnet.preprocess_input
      base_model = tf.keras.applications.ResNet101
    elif pretrained_arch == 'resnet50v2':
      preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
      base_model = tf.keras.applications.ResNet50V2
    elif pretrained_arch == 'resnet101v2':
      preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
      base_model = tf.keras.applications.ResNet101V2
    elif pretrained_arch == 'inceptionv3':
      preprocess_input = tf.keras.applications.inception_v3.preprocess_input
      base_model = tf.keras.applications.InceptionV3
    elif pretrained_arch == 'inceptionresnetv2':
      preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
      base_model = tf.keras.applications.InceptionResNetV2
    elif pretrained_arch == 'densenet169':
      preprocess_input = tf.keras.applications.densenet.preprocess_input
      base_model = tf.keras.applications.DenseNet169
    elif pretrained_arch == 'densenet201':
      preprocess_input = tf.keras.applications.densenet.preprocess_input
      base_model = tf.keras.applications.DenseNet201
    elif pretrained_arch == 'efficientnetb0':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB0
    elif pretrained_arch == 'efficientnetb1':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB1
    elif pretrained_arch == 'efficientnetb2':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB2
    elif pretrained_arch == 'efficientnetb3':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB3
    elif pretrained_arch == 'efficientnetb4':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB4
    elif pretrained_arch == 'efficientnetb5':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB5
    elif pretrained_arch == 'efficientnetb6':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB6
    elif pretrained_arch == 'efficientnetb7':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB7
    return base_model, preprocess_input

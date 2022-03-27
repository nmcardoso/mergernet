import logging
import secrets
import shutil
from typing import Any, Dict, Tuple, Union
from pathlib import Path

import optuna
import mlflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from optuna.integration.mlflow import MLflowCallback

from mergernet.core.constants import MLFLOW_DEFAULT_DB, MLFLOW_DEFAULT_URL, RANDOM_SEED, SAVED_MODELS_PATH
from mergernet.core.dataset import Dataset
from mergernet.core.entity import ConstantHyperParameter, HyperParameterSet
from mergernet.model.callback import DeltaStopping
from mergernet.model.plot import conf_matrix
from mergernet.model.preprocessing import load_jpg, load_png, one_hot_factory
from mergernet.core.utils import Timming


L = logging.getLogger('job')

# this implementation mimics the mlflow integration of optuna
# `RUN_ID_ATTRIBUTE_KEY` was extracted from following code:
# https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
RUN_ID_ATTRIBUTE_KEY = 'mlflow_run_id'



class PruneCallback(tf.keras.callbacks.Callback):
  def __init__(self, trial: optuna.trial.FrozenTrial, objective_metric: str):
    super(PruneCallback, self).__init__()
    self.trial = trial
    self.objective_metric = objective_metric # default: "val_loss"


  def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
    self.trial.report(value=logs[self.objective_metric], step=epoch)

    if self.trial.should_prune():
      self.model.stop_training = True
      L.info(f'[PRUNER] trial pruned at epoch {epoch + 1}')


class SaveCallback(tf.keras.callbacks.Callback):
  def __init__(
    self,
    name: str,
    study: optuna.study.Study,
    objective_metric: str,
    objective_direction: str
  ):
    super(SaveCallback, self).__init__()
    self.name = name
    self.study = study
    self.objective_metric = objective_metric
    self.default_value = -np.inf if objective_direction == 'maximize' else np.inf
    self.operator = np.greater if objective_direction == 'maximine' else np.less


  def on_train_end(self, logs: Dict[str, Any]):
    try:
      best_value = self.study.best_value
    except:
      best_value = self.default_value

    if self.operator(logs[self.objective_metric], best_value):
      save_path = SAVED_MODELS_PATH / (self.name + '.h5')
      if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
      self.model.save(save_path, overwrite=True)


class MLflowTensorflowCallback(tf.keras.callbacks.Callback):
  def __init__(
    self,
    trial: optuna.trial.FrozenTrial,
    dataset: Dataset,
    ds_test: tf.data.Dataset
  ):
    self.trial = trial
    self.dataset = dataset
    self.ds_test = ds_test


  def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
    with mlflow.start_run(run_name=str(self.trial.number), nested=self.nest_trials) as run:
      # The mlflow run will be created before optuna mlflow callback,
      # so the following line is needed in order to optuna get the current run.
      # https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
      self.trial.set_system_attr(RUN_ID_ATTRIBUTE_KEY, run.info.run_id)

      mlflow.log_metrics(metric=logs, step=epoch)


  def on_train_end(self, logs: Dict[str, Any]):
    with mlflow.start_run(run_name=str(self.trial.number), nested=self.nest_trials) as run:
      # The mlflow run will be created before optuna mlflow callback,
      # so the following line is needed in order to optuna get the current run.
      # https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
      self.trial.set_system_attr(RUN_ID_ATTRIBUTE_KEY, run.info.run_id)

      # confusion matrix plot
      y_pred = self.model.predict(self.ds_test)
      y_true = np.concatenate([y for _, y in self.ds_test], axis=0)
      lm = self.dataset.config.label_map
      labels = [[*lm.keys()][v] for v in lm.values()]
      ax = conf_matrix(y_true, y_pred, one_hot=True, labels=labels)
      mlflow.log_figure(ax.figure, f'confusion_matrix.png')
      plt.close(ax.figure)

      # log predictions
      mlflow.log_dict(
        {
          'dataset': self.dataset.config.name,
          'X': np.array(self.dataset.get_X_by_fold(0, kind='test')).tolist(),
          'y_pred': np.array(y_pred).tolist(),
          'y_true': np.array(y_true).tolist()
        },
        'predictions.json'
      )



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
    self.hp = None
    self.study = None
    self.save_model = None
    self.mlflow_enabled = None
    self.objective_metric = 'val_loss'
    self.objective_direction = 'minimize'


  def prepare_data(
    self,
    ds: tf.data.Dataset,
    batch_size: int = 64,
    buffer_size: int = 1000,
    kind='train'
  ):
    if self.dataset.config.X_column_suffix == '.jpg':
      ds = ds.map(load_jpg)
      L.info('[DATASET] apply: load_jpg')
    elif self.dataset.config.X_column_suffix == '.png':
      ds = ds.map(load_png)
      L.info('[DATASET] apply: load_png')

    if kind == 'train':
      ds = ds.map(one_hot_factory(self.dataset.config.n_classes))
      L.info('[DATASET] apply: one_hot')

    ds = ds.cache()
    L.info('[DATASET] apply: cache')

    ds = ds.shuffle(buffer_size)
    L.info('[DATASET] apply: shuffle')

    ds = ds.batch(batch_size)
    L.info('[DATASET] apply: batch')

    ds = ds.prefetch(tf.data.AUTOTUNE)
    L.info('[DATASET] apply: prefetch')

    return ds


  def build_model(
    self,
    input_shape: Tuple,
    trial: optuna.trial.FrozenTrial = None,
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
    model.compile(
      optimizer=tf.keras.optimizers.Adam(self.hp.learning_rate.suggest()),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
      ]
    )
    return model


  def predict(
    self,
    model_name: str,
    model: Union[str, Path, tf.keras.Model],
    dataset: Dataset
  ) -> list:
    tf.keras.backend.clear_session()

    if isinstance(model, (str, Path)):
      model = tf.keras.models.load_model(model)

    ds = dataset.get_preds_dataset()
    ds = self.prepare_data(ds, batch_size=128, buffer_size=1000, kind='predict')

    preds = model.predict(ds)

    with mlflow.start_run(run_name='predict', nested=self.nest_trials) as run:
      mlflow.log_params({
        'model': model_name,
        'dataset': self.dataset.config.name
      })
      # mlflow.log_params(
      #   {
      #     hp_name: hp_instance.value
      #     for hp_name, hp_instance in self.hp.__dict__.items()
      #     if not hp_name.startswith('_') and isinstance(hp_instance, ConstantHyperParameter)
      #   }
      # )

      mlflow.log_dict(
        {
          'dataset': self.dataset.config.name,
          'X': np.array(self.dataset.get_X()).tolist(),
          'y_pred': np.array(preds).tolist()
        },
        'predictions.json'
      )

    return preds


  def objective(self, trial: optuna.trial.FrozenTrial) -> float:
    tf.keras.backend.clear_session()

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
    class_weights = self.dataset.compute_class_weight()

    model = self.build_model(
      input_shape=self.dataset.config.image_shape,
      trial=trial
    )

    callbacks = [
      PruneCallback(trial=trial, objective_metric=self.objective_metric)
    ]
    if self.mlflow_enabled:
      callbacks.append(
        MLflowTensorflowCallback(
          trial=trial,
          dataset=self.dataset,
          ds_test=ds_test
        )
      )
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
    L.info('[TRAIN] Start of training loop')
    history = model.fit(
      ds_train,
      batch_size=self.hp.batch_size.suggest(trial),
      epochs=self.epochs,
      validation_data=ds_test,
      class_weight=class_weights,
      callbacks=callbacks
    )
    L.info(f'[TRAIN] End of training loop, duration: {t.end()}.')

    # history shape: {metric1: [val1, val2, ...], metric2: [val1, val2, ...]}
    h = history.history
    # epochs = len(h[list(h.keys())[0]])

    # if self.mlflow_enabled:
    #   with mlflow.start_run(run_name=str(trial.number), nested=self.nest_trials) as run:
    #     # The mlflow run will be created before optuna mlflow callback,
    #     # so the following line is needed in order to optuna get the current run.
    #     # https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
    #     trial.set_system_attr(RUN_ID_ATTRIBUTE_KEY, run.info.run_id)

    #     for i in range(epochs):
    #       # {k1: [v1, v2, ...], k2: [v1, v2, ...]} => {k1: vi, k2: vi}
    #       metrics = {name: h[name][i] for name in h.keys()}
    #       mlflow.log_metrics(metrics=metrics, step=i)

    #     # confusion matrix plot
    #     y_pred = model.predict(ds_test)
    #     y_true = np.concatenate([y for x, y in ds_test], axis=0)
    #     lm = self.dataset.config.label_map
    #     labels = [[*lm.keys()][v] for v in lm.values()]
    #     ax = conf_matrix(y_true, y_pred, one_hot=True, labels=labels)
    #     mlflow.log_figure(ax.figure, f'confusion_matrix.png')
    #     plt.close(ax.figure)

    #     # log predictions
    #     mlflow.log_dict(
    #       {
    #         'dataset': self.dataset.config.name,
    #         'X': np.array(self.dataset.get_X_by_fold(0, kind='test')).tolist(),
    #         'y_pred': np.array(y_pred).tolist(),
    #         'y_true': np.array(y_true).tolist()
    #       },
    #       'predictions.json'
    #     )

    # generating optuna value to optimize (val_accuracy)
    last_epoch_accuracy = h[self.objective_metric][-1]
    return last_epoch_accuracy



  def train(
    self,
    hyperparameters: HyperParameterSet,
    save_path: Union[str, Path] = None
  ):
    self.hp = hyperparameters

    tf.keras.backend.clear_session()

    ds_train, ds_test = self.dataset.get_fold(0)
    ds_train = self.prepare_data(
      ds_train,
      batch_size=self.hp.batch_size.suggest(),
      buffer_size=5000,
      kind='train'
    )
    ds_test = self.prepare_data(
      ds_test,
      batch_size=self.hp.batch_size.suggest(),
      buffer_size=1000,
      kind='train'
    )
    class_weights = self.dataset.compute_class_weight()

    model = self.build_model(input_shape=self.dataset.config.image_shape)

    t = Timming()
    L.info('[TRAIN] Start of training loop')
    history = model.fit(
      ds_train,
      batch_size=self.hp.batch_size.suggest(),
      epochs=self.epochs,
      validation_data=ds_test,
      class_weight=class_weights,
      callbacks=[]
    )
    L.info(f'[TRAIN] End of training loop, duration: {t.end()}.')

    with mlflow.start_run(run_name='predict', nested=self.nest_trials) as run:
      pass


  def hypertrain(
    self,
    optuna_uri: str,
    n_trials: int,
    hyperparameters: HyperParameterSet,
    pruner: str = 'hyperband',
    objective_metric: str = 'val_loss',
    objective_direction: str = 'minimize',
    resume: bool = False,
    save_model: bool = True,
    mlflow_enabled: bool = True
  ):
    self.hp = hyperparameters
    self.save_model = save_model
    self.mlflow_enabled = mlflow_enabled
    self.objective_metric = objective_metric
    self.objective_direction = objective_direction

    if resume:
      L.info(f'[HYPER] resuming previous optimization of {self.name} study')

    if pruner == 'median':
      pruner_instance = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10
      )
    elif pruner == 'hyperband':
      pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

    L.info(f'[HYPER] start of optuna optimization')

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

    if self.mlflow_enabled:
      mlflow_cb = MLflowCallback(
        metric_name='optuna_score',
        nest_trials=self.nest_trials,
        tag_study_user_attrs=False
      )
      optimize_params['callbacks'] = mlflow_cb

    study.optimize(**optimize_params)

    L.info(f'[HYPER] optuna optimization finished in {t.end()}')
    L.info(f'[HYPER] number of finished trials: {len(study.trials)}')
    L.info(f'[HYPER] ----- begin of best trial summary -----')
    L.info(f'[HYPER] optimization score: {study.best_trial.value}')
    L.info(f'[HYPER] params:')
    for k, v in study.best_params.items():
      L.info(f'[HYPER] {k}: {str(v)}')
    L.info(f'[HYPER] ----- end of best trial summary -----')


  def _architecture_switch(self, pretrained_arch):
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

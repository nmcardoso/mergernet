import logging
import secrets
import shutil
from typing import Tuple

import optuna
import mlflow
import tensorflow as tf
import numpy as np
from mergernet.core.artifacts import ArtifactHelper
from optuna.integration.mlflow import MLflowCallback
from mergernet.core.constants import MLFLOW_DEFAULT_DB, MLFLOW_DEFAULT_URL, RANDOM_SEED

from mergernet.core.dataset import Dataset
from mergernet.core.entity import HyperParameterSet
from mergernet.model.callback import DeltaStopping
from mergernet.model.plot import conf_matrix
from mergernet.model.preprocessing import load_jpg, one_hot
from mergernet.core.utils import Timming


L = logging.getLogger('job')

# this implementation mimics the mlflow integration of optuna
# `RUN_ID_ATTRIBUTE_KEY` was extracted from following code:
# https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
RUN_ID_ATTRIBUTE_KEY = 'mlflow_run_id'


class PruneCallback(tf.keras.callbacks.Callback):
  def __init__(self, trial: optuna.trial.FrozenTrial):
    super(PruneCallback, self).__init__()
    self.trial = trial


  def on_epoch_end(self, epoch, logs):
    self.trial.report(value=logs['val_accuracy'], step=epoch)

    if self.trial.should_prune():
      self.model.stop_training = True
      L.info(f'[PRUNER] trial pruned at epoch {epoch + 1}')



class HyperModel:
  def __init__(
    self,
    dataset: Dataset,
    name: str,
    hyperparameters: HyperParameterSet,
    epochs: int = 20,
    nest_trials: bool = False
  ):
    self.dataset = dataset
    self.name = name
    self.hp = hyperparameters
    self.epochs = epochs
    self.nest_trials = nest_trials

    self.mlflow_cb = MLflowCallback(
      metric_name='optuna_score',
      nest_trials=self.nest_trials,
      tag_study_user_attrs=False
    )


  def prepare_data(self, dataset: Dataset):
    ds_train, ds_test = dataset.get_fold(0)
    L.info('[DATASET] Fold 0 loaded')

    if not dataset.in_memory:
      ds_train = ds_train.map(load_jpg)
      ds_test = ds_test.map(load_jpg)
      L.info('[DATASET] apply: load_jpg')

    ds_train = ds_train.map(one_hot)
    ds_test = ds_test.map(one_hot)
    L.info('[DATASET] apply: one_hot')

    ds_train = ds_train.cache()
    ds_test = ds_test.cache()

    ds_train = ds_train.shuffle(5000)
    ds_test = ds_test.shuffle(1000)
    L.info('[DATASET] apply: shuffle')

    ds_train = ds_train.batch(64)
    ds_test = ds_test.batch(64)
    L.info('[DATASET] apply: batch')
    _x, _y = next(ds_test.take(1).as_numpy_iterator())
    print('X.shape =', _x.shape, 'y.shape =', _y.shape)

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train
    ds_test = ds_test
    class_weights = dataset.compute_class_weight()

    return ds_train, ds_test, class_weights



  def build_model(
    self,
    trial: optuna.trial.FrozenTrial,
    input_shape: Tuple
  ):
    self.hp.set_trial(trial)

    conv_block = tf.keras.applications.ResNet50(
      input_shape=input_shape,
      include_top=False,
      weights=self.hp.pretrained_weights.suggest()
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


  def objective(self, trial: optuna.trial.FrozenTrial):
    tf.keras.backend.clear_session()

    ds_train, ds_test, class_weights = self.prepare_data(self.dataset)

    model = self.build_model(
      trial,
      input_shape=self.dataset.config.image_shape
    )

    t = Timming()
    L.info('[TRAIN] Start of training loop')

    history = model.fit(
      ds_train,
      batch_size=self.hp.batch_size.suggest(trial),
      epochs=self.epochs,
      validation_data=ds_test,
      class_weight=class_weights,
      callbacks=[
        PruneCallback(trial)
      ]
    )

    L.info(f'[TRAIN] End of training loop, duration: {t.end()}.')

    # mlflow logs
    with mlflow.start_run(run_name=str(trial.number), nested=self.nest_trials) as run:
      # The mlflow run will be created before optuna mlflow callback,
      # so the following line is needed in order to optuna get the current run.
      # https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
      trial.set_system_attr(RUN_ID_ATTRIBUTE_KEY, run.info.run_id)

      # history shape: {metric1: [val1, val2, ...], metric2: [val1, val2, ...]}
      h = history.history
      epochs = len(h[list(h.keys())[0]])

      for i in range(epochs):
        # {k1: [v1, v2, ...], k2: [v1, v2, ...]} => {k1: vi, k2: vi}
        metrics = {name: h[name][i] for name in h.keys()}
        mlflow.log_metrics(metrics=metrics, step=i)

      # confusion matrix plot
      y_pred = model.predict(ds_test)
      y_true = np.concatenate([y for x, y in ds_test], axis=0)
      ax = conf_matrix(y_true, y_pred, one_hot=True)
      mlflow.log_figure(ax.figure, f'confusion_matrix_{trial.number}.png')

    # generating optuna value to optimize (val_accuracy)
    last_epoch_accuracy = h['val_accuracy'][-1]
    ev = model.evaluate(ds_test)
    idx = model.metrics_names.index('accuracy')
    print('last_epoch_acc', last_epoch_accuracy, 'eval', ev[idx])

    return last_epoch_accuracy #ev[idx]



  def hypertrain(self, optuna_uri: str, n_trials: int, pruner: str = 'hyperband', resume: bool = False):
    if resume:
      L.info(f'[HYPER] resuming previous optimization of {self.name} study')

    if pruner == 'median':
      pruner_instance = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10
      )
    elif pruner == 'hyperband':
      pruner_instance = optuna.pruners.HyperbandPruner(min_resource=4)

    L.info(f'[HYPER] start of optuna optimization')

    t = Timming()
    study = optuna.create_study(
      storage=optuna_uri,
      study_name=self.name,
      pruner=pruner_instance,
      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
      direction='maximize',
      load_if_exists=resume
    )
    study.optimize(
      self.objective,
      n_trials=n_trials,
      callbacks=[self.mlflow_cb]
    )

    L.info(f'[HYPER] optuna optimization finished in {t.end()}')

    L.info(f'[HYPER] number of finished trials: {len(study.trials)}')
    L.info(f'[HYPER] ----- begin of best trial summary -----')
    L.info(f'[HYPER] optimization score: {study.best_trial.value}')
    L.info(f'[HYPER] params:')
    for k, v in study.best_params.items():
      L.info(f'[HYPER] {k}: {str(v)}')
    L.info(f'[HYPER] ----- end of best trial summary -----')

    # ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    # fig = optuna.visualization.plot_optimization_history(study)
    # mlflow.log_figure(fig, 'optimization_history.html')
    # path = str((ah.artifact_path / 'optimization_history.png').resolve())
    # ax.figure.savefig(path,  pad_inches=0.01, bbox_inches='tight')
    # ah.upload(fname='optimization_history.png', github=True, gdrive=True)

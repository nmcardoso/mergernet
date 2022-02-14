import logging
import secrets
import shutil

import optuna
import mlflow
import tensorflow as tf
from mergernet.core.artifacts import ArtifactHelper
from optuna.integration.mlflow import MLflowCallback
from mergernet.core.constants import MLFLOW_DEFAULT_DB, RANDOM_SEED

from mergernet.core.dataset import Dataset
from mergernet.model.callback import DeltaStopping
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
    resume: bool = False,
    default_mlflow: bool = True,
    nest_trials: bool = False
  ):
    ah = ArtifactHelper()

    self.optuna_uri = f'sqlite:///{str(ah.artifact_path.resolve())}/optuna/{name}.sqlite'
    ah.optuna_db_name = name

    if default_mlflow:
      self.mlflow_uri = f'sqlite:///{str(ah.artifact_path.resolve())}/mlflow/{MLFLOW_DEFAULT_DB}'
    else:
      self.mlflow_uri = f'sqlite:///{str(ah.artifact_path.resolve())}/mlflow/{name}.sqlite'

    print(self.mlflow_uri)
    print(self.optuna_uri)

    self.dataset = dataset
    self.name = name
    self.resume = resume
    self.nest_trials = nest_trials

    self.mlflow_cb = MLflowCallback(
      metric_name='optuna_score',
      nest_trials=self.nest_trials,
      tag_study_user_attrs=False
    )


  def prepare_data(self, dataset):
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

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train
    ds_test = ds_test
    class_weights = dataset.compute_class_weight()

    return ds_train, ds_test, class_weights



  def build_model(self, trial, input_shape, pretrained_weights):
    conv_block = tf.keras.applications.ResNet50(
      input_shape=input_shape,
      include_top=False,
      weights=pretrained_weights
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
    x = tf.keras.layers.Dense(trial.suggest_categorical('dense_units_1', [64, 128, 256, 512, 1024]), activation='relu')(x)
    x = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_rate_1', 0.1, 0.5))(x)
    x = tf.keras.layers.Dense(trial.suggest_categorical('dense_units_2', [64, 128, 256, 512, 1024]), activation='relu')(x)
    x = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_rate_2', 0.1, 0.5))(x)

    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
      optimizer=tf.keras.optimizers.Adam(
        trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
      ),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
      ]
    )
    return model


  def objective(self, trial):
    tf.keras.backend.clear_session()

    batch_size = 64

    ds_train, ds_test, class_weights = self.prepare_data(self.dataset)

    model = self.build_model(
      trial,
      input_shape=(128, 128, 3),
      pretrained_weights='imagenet',
    )

    t = Timming()
    t.start()
    L.info('[TRAIN] Starting training loop')

    ah = ArtifactHelper()

    history = model.fit(
      ds_train,
      batch_size=batch_size,
      epochs=self.epochs,
      validation_data=ds_test,
      class_weight=class_weights,
      callbacks=[
        # tf.keras.callbacks.EarlyStopping(patience=3),
        # DeltaStopping(),
        PruneCallback(trial)
      ]
    )

    t.end()
    L.info(f'[TRAIN] Training finished without errors in {t.duration()}.')


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

    ev = model.evaluate(ds_test)
    idx = model.metrics_names.index('accuracy')
    return ev[idx]



  def hypertrain(self, n_trials: int, epochs: int, pruner: str = 'hyperband'):
    # mlflow must be initialized here, not by the callback
    mlflow.set_tracking_uri(self.mlflow_uri)
    mlflow.set_experiment(self.name)

    self.epochs = epochs

    ah = ArtifactHelper()
    optuna_path = ah.artifact_path / 'optuna' / f'{self.name}.sqlite'

    if self.resume:
      L.info(f'[HYPER] resuming previous optimization of {self.name} study')

    if self.resume and not optuna_path.exists():
      ah.download_optuna_db(self.name)
    elif not self.resume and optuna_path.exists():
      optuna_path.unlink()

    if pruner == 'median':
      pruner_instance = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    elif pruner == 'hyperband':
      pruner_instance = optuna.pruners.HyperbandPruner(min_resource=4)

    L.info(f'[HYPER] start of optuna optimization')
    t = Timming()
    t.start()

    study = optuna.create_study(
      storage=self.optuna_uri,
      study_name=self.name,
      pruner=pruner_instance,
      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
      direction='maximize',
      load_if_exists=self.resume
    )
    study.optimize(self.objective, n_trials=n_trials, callbacks=[self.mlflow_cb])

    t.end()
    L.info(f'[HYPER] optuna optimization finished in {t.duration()}')

    L.info(f'[HYPER] number of finished trials: {len(study.trials)}')
    L.info(f'[HYPER] ----- begin of best trial summary -----')
    L.info(f'[HYPER] optimization score: {study.best_trial.value}')
    L.info(f'[HYPER] params:')
    for k, v in study.best_params.items():
      L.info(f'[HYPER] {k}: {str(v)}')
    L.info(f'[HYPER] ----- end of best trial summary -----')

    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    path = str((ah.artifact_path / 'optimization_history.png').resolve())
    ax.figure.savefig(path,  pad_inches=0.01, bbox_inches='tight')
    ah.upload(fname='optimization_history.png', github=True, gdrive=True)

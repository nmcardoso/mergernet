import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import optuna
import tensorflow as tf
import wandb

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.estimators.base import Estimator
from mergernet.model.utils import history_to_dataframe, setup_seeds

setup_seeds()

L = logging.getLogger(__name__)


class TrainStrategy(ABC):
  def __init__(
    self,
    model: Estimator,
    dataset: Dataset,
    hp: HyperParameterSet,
    callbacks: List[tf.keras.Callback]
  ):
    self.model = model
    self.dataset = dataset
    self.hp = hp
    self.callbacks = callbacks

  @abstractmethod
  def train(self, run_name: str = '') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    pass

  def set_trainable(self, tf_model: tf.keras.Model, layer: str, trainable: bool):
    for l in tf_model.layers:
      if l.name == layer:
        l.trainable = trainable

  def compile_model(
    self,
    tf_model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    metrics: list = []
  ):
    tf_model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        *metrics
      ]
    )


class ParametricStrategy(TrainStrategy):
  def __init__(
    self,
    model: Estimator,
    dataset: Dataset,
    hp: HyperParameterSet,
    callbacks=List[tf.keras.Callback],
  ):
    super().__init__(model, dataset, hp, callbacks)

  def train(self, run_name: str = 'run-0') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
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

    model = self.model.build(
      input_shape=self.dataset.config.image_shape,
      n_classes=self.dataset.config.n_classes,
      freeze_conv=True,
      hp=self.hp
    )
    self.compile_model(model, tf.keras.optimizers.Adam(self.hp.get('opt_lr')))

    with Experiment.Tracer(self.hp.to_values_dict(), name=run_name, job_type='train'):
      early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        mode='min', # 'min' or 'max'
        restore_best_weights=True
      )

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
        batch_size=self.hp.get('batch_size'),
        epochs=self.hp.get('tl_epochs', default=10),
        validation_data=ds_test,
        class_weight=class_weights,
        callbacks=[early_stop_cb, wandb_cb]
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

    return model


class OptunaStrategy(TrainStrategy):
  def __init__(
    self,
    train_strategy: TrainStrategy,
    n_trials: int,
    pruner: str = 'hyperband',
    objective_metric: str = 'val_loss',
    objective_direction: str = 'minimize',
    resume_hash: str = None
  ):
    super().__init__(
      train_strategy.model,
      train_strategy.dataset,
      train_strategy.hp,
      train_strategy.callbacks
    )
    self.train_strategy = train_strategy
    self.n_trials = n_trials
    self.pruner = pruner
    self.objective_metric = objective_metric
    self.objective_direction = objective_direction
    self.resume_hash = resume_hash
    self.study = None


  def _objective(self, trial: optuna.trial.FrozenTrial) -> float:
    L.info(f'Starting trial {trial.number}')

    # generate callback
    try:
      initial_value = self.study.best_value
    except:
      initial_value = None

    model_path = Experiment.local_exp_path / 'model.h5'
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
      str(model_path),
      monitor='val_loss',
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='min',
      save_freq='epoch',
      initial_value_threshold=initial_value
    )
    L.info(f'Initial threshold for CheckpointCallback: {initial_value}')

    # run train function
    self.hp.set_trial(trial)
    self.train_strategy.callbacks.append(ckpt_callback)
    model = self.train_strategy.train(run_name=f'run-{trial.number}')

    hist = model.history.history
    # saving the history of current trial
    hist_df = history_to_dataframe(hist)
    Experiment.upload_file_gd(f'history_trial_{trial.number}.csv', hist_df)

    # generating optuna value to optimize (val_accuracy)
    objective_value = hist['val_loss'][-1]

    return objective_value


  def train(self, run_name: str = 'run-0') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    exp_id = Experiment.exp_id

    # get prunner
    if self.pruner == 'median':
      pruner_instance = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10
      )
    elif self.pruner == 'hyperband':
      pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

    # set db uri
    optuna_path = Experiment.local_exp_path / 'optuna.sqlite'
    optuna_uri = f'sqlite:///{str(optuna_path.resolve())}' # absolute path

    if self.resume_hash is not None:
      L.info(f'Downloading optuna study of exp {exp_id}')
      Experiment.download_file_gh('optuna.sqlite', exp_id)

    # creating a new study instance
    if self.resume_hash is None:
      study_factory = optuna.create_study
    else:
      study_factory = optuna.load_study

    study = study_factory(
      storage=optuna_uri,
      study_name=Experiment.exp_name,
      pruner=pruner_instance,
      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
      direction=self.objective_direction,
      load_if_exists=True
    )

    # start optimization (train loop)
    L.info(f'start of optuna optimization')
    t = Timming()
    study.optimize(func=self._objective, n_trials=self.n_trials)
    L.info(f'optuna optimization finished in {t.end()}')

    # save optimization artifacts
    Experiment.upload_file_gd('model.h5')
    Experiment.upload_file_gd('optuna.sqlite')
    Experiment.upload_file_gd(
      'optuna_trials.csv',
      study.trials_dataframe(multi_index=False)
    )

    L.info(f'number of finished trials: {len(study.trials)}')
    L.info(f'----- begin of best trial summary -----')
    L.info(f'optimization score: {study.best_trial.value}')
    L.info(f'params:')
    for k, v in study.best_params.items():
      L.info(f'{k}: {str(v)}')
    L.info(f'----- end of best trial summary -----')

    # load model and return
    model = tf.keras.models.load_model(Experiment.local_exp_path / 'model.h5')
    return model


class ZoobotStrategy(TrainStrategy):
  def __init__(
    self,
    model: Estimator,
    dataset: Dataset,
    hp: HyperParameterSet,
    callbacks=List[tf.keras.Callback],
  ):
    super().__init__(model, dataset, hp, callbacks)

  def train(self, run_name: str = 'run-0') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    from zoobot.tensorflow.estimators import define_model
    tf_model = define_model.get_model()

  def predict(self):
    pass

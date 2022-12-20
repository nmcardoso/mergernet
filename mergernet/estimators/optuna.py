import logging
from pathlib import Path
from typing import Tuple

import optuna
import tensorflow as tf
import wandb

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.estimators.base import Estimator
from mergernet.model.utils import history_to_dataframe

L = logging.getLogger(__name__)


MODEL_FILENAME = 'model.h5'
OPTUNA_DB_FILENAME = 'optuna.sqlite'


class OptunaEstimator(Estimator):
  def __init__(
    self,
    hp: HyperParameterSet,
    dataset: Dataset,
    estimator: Estimator,
    n_trials: int,
    pruner: str,
    objective_metric: str,
    objective_direction: str,
    resume_hash: str
  ):
    super().__init__(hp, dataset)
    self.estimator = estimator
    self.n_trials = n_trials
    self.pruner = pruner
    self.objective_metric = objective_metric
    self.objective_direction = objective_direction
    self.resume_hash = resume_hash
    self.study = None


  def _objective(self, trial: optuna.trial.Trial) -> float:
    """
    Objective function that will be optimized by oputna

    Parameters
    ----------
    trial: optuna.trial.Trial
      The current optuna trial

    Returns
    -------
    float
      The metric that will be optimized by optuna
    """
    L.info(f'Starting trial {trial.number}')

    # generate callback
    try:
      initial_value = self.study.best_value
    except:
      initial_value = None

    model_path = Path(Experiment.local_exp_path) / MODEL_FILENAME
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

    # update current trial in hyperparameter set
    self.hp.set_trial(trial)
    self.hp.clear_values_dict()

    # train model
    model = self.estimator.train(
      run_name=f'run-{trial.number}',
      callbacks=[ckpt_callback]
    )

    # saving the history of current trial
    hist = model.history.history
    hist_df = history_to_dataframe(hist)
    Experiment.upload_file_gd(f'history_trial_{trial.number}.csv', hist_df)

    # generating optuna value to optimize (val_accuracy)
    objective_value = hist['val_loss'][-1]

    return objective_value


  def train(self, *args, **kwargs) -> Tuple[tf.keras.Model, tf.keras.History]:
    """
    Starts the optuna optimization and returns the best model trained

    Returns
    -------
    tf.keras.Model
      The best model trained
    """
    exp_id = Experiment.exp_id

    # get prunner
    if self.pruner == 'median':
      pruner_instance = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10
      )
    elif self.pruner == 'hyperband':
      pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

    # setup db uri
    optuna_path = Path(Experiment.local_exp_path) / OPTUNA_DB_FILENAME
    optuna_uri = f'sqlite:///{str(optuna_path.resolve())}' # absolute path

    if self.resume_hash is not None:
      L.info(f'Downloading optuna study of exp {exp_id}')
      Experiment.download_file_gh(OPTUNA_DB_FILENAME, exp_id)

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

    # load and return the best saved model
    self._tf_model = tf.keras.models.load_model(
      Path(Experiment.local_exp_path) / MODEL_FILENAME
    )
    return self._tf_model

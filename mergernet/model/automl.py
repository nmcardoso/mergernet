import logging
from pathlib import Path
from types import FunctionType

import optuna
import tensorflow as tf

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.model.utils import history_to_dataframe

L = logging.getLogger(__name__)
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()



def _objective_factory(
  train_func: FunctionType,
  dataset: Dataset,
  hp: HyperParameterSet,
  study: optuna.study.Study
) -> FunctionType:
  def objective(trial: optuna.trial.FrozenTrial) -> float:
    L.info(f'Starting trial {trial.number}')

    # generate callback
    try:
      initial_value = study.best_value
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
    hp.set_trial(trial)
    model = train_func(
      dataset=dataset,
      hp=hp,
      callbacks=[ckpt_callback],
      run_name=f'run-{trial.number}'
    )

    hist = model.history.history
    # saving the history of current trial
    hist_df = history_to_dataframe(hist)
    Experiment.upload_file_gd(f'history_trial_{trial.number}.csv', hist_df)

    # generating optuna value to optimize (val_accuracy)
    objective_value = hist['val_loss'][-1]

    return objective_value
  return objective



def optuna_train(
  train_func: FunctionType,
  dataset: Dataset,
  hp: HyperParameterSet,
  n_trials: int,
  pruner: str = 'hyperband',
  objective_metric: str = 'val_loss',
  objective_direction: str = 'minimize',
  resume_hash: str = None
):
  exp_id = Experiment.exp_id

  # get prunner
  if pruner == 'median':
    pruner_instance = optuna.pruners.MedianPruner(
      n_startup_trials=10,
      n_warmup_steps=10
    )
  elif pruner == 'hyperband':
    pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

  # set db uri
  optuna_path = Experiment.local_exp_path / 'optuna.sqlite'
  optuna_uri = f'sqlite:///{str(optuna_path.resolve())}' # absolute path

  if resume_hash is not None:
    L.info(f'Downloading optuna study of exp {exp_id}')
    Experiment.download_file_gh('optuna.sqlite', exp_id)

  # creating a new study instance
  if resume_hash is None:
    study_factory = optuna.create_study
  else:
    study_factory = optuna.load_study

  study = study_factory(
    storage=optuna_uri,
    study_name=Experiment.exp_name,
    pruner=pruner_instance,
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    direction=objective_direction,
    load_if_exists=True
  )

  # start optimization (train loop)
  L.info(f'start of optuna optimization')
  t = Timming()
  study.optimize(
    func=_objective_factory(train_func, dataset, hp, study),
    n_trials=n_trials
  )
  t.end()
  L.info(f'optuna optimization finished in {t.duration()}')

  # save optimization artifacts
  Experiment.upload_file_gd('model.h5')
  Experiment.upload_file_gd('optuna.sqlite')
  Experiment.upload_file_gd('optuna_trials.csv', study.trials_dataframe(multi_index=False))

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

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

from mergernet.core.experiment import Experiment

L = logging.getLogger(__name__)



class DeltaStopping(tf.keras.callbacks.Callback):
  def __init__(self):
    super(DeltaStopping, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    if (epoch > 2) and (logs['accuracy'] - logs['val_accuracy']) > 0.1:
      self.model.stop_training = True



class PruneCallback(tf.keras.callbacks.Callback):
  def __init__(self, trial: optuna.trial.FrozenTrial, objective_metric: str):
    super(PruneCallback, self).__init__()
    self.trial = trial
    self.objective_metric = objective_metric # default: "val_loss"


  def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
    self.trial.report(value=logs[self.objective_metric], step=epoch)

    if self.trial.should_prune():
      self.model.stop_training = True
      L.info(f'Trial pruned at epoch {epoch + 1}')



class SaveBestTrialCallback(tf.keras.callbacks.Callback):
  def __init__(
    self,
    study: optuna.study.Study,
    objective_metric: str,
    objective_direction: str,
    name: str = 'model'
  ):
    super(SaveBestTrialCallback, self).__init__()
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

    current_value = logs[self.objective_metric]
    if self.operator(current_value, best_value):
      L.info(f'New best metric detected. {self.objective_metric}: {current_value}')

      # save model
      save_path = Path(Experiment.local_run_path) / f'{self.name}.h5'
      self.model.save(save_path, overwrite=True)
      L.info(f'Trial saved in {str(save_path)}')
      Experiment.register_artifact(f'{self.name}.h5', 'gdrive')

      # save history as csv
      hist = self.model.history.history.copy()
      hist['epoch'] = range(len(hist['loss']))
      hist_df = pd.DataFrame(hist)
      save_path = Path(Experiment.local_run_path) / f'history_{self.name}.json'
      hist_df.to_csv(save_path, index=False)
      L.info(f'History saved in {str(save_path)}')
      Experiment.register_artifact(f'history_{self.name}.json', 'github')



class TelemetryCallback(tf.keras.callbacks.Callback):
  def __init__(self, trial: optuna.trial.FrozenTrial):
    super(TelemetryCallback, self).__init__()
    self.trial = trial

  def on_train_end(self, logs: Dict[str, Any]):
    pass

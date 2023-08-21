from typing import Dict, List

import pandas as pd
import tensorflow as tf
import wandb

from mergernet.core.experiment import Experiment
from mergernet.core.utils import load_table
from mergernet.data.dataset import Dataset


class Predictor:
  def __init__(self, model: tf.keras.Model, dataset: Dataset):
    self.model = model
    self.dataset = dataset
    self._preds = None


  def predict(self) -> List:
    # select ds by dataset type: predictions or train
    if self.dataset.config.label_column is None:
      ds_test = self.dataset.get_preds_dataset()
    else:
      _, ds_test = self.dataset.get_fold(0)

    # prepare dataset
    ds_test = self.dataset.prepare_data(ds_test, kind='pred')

    # make prediction and save in instance state
    test_preds = self.model.predict(ds_test)
    self._preds = test_preds

    return test_preds


  def upload(self, name: str = None, labels: Dict = None):
    # get X by dataset type: predictions or train
    if self.dataset.config.label_column is None:
      X = self.dataset.get_X()
      name = name or 'predictions.csv'
    else:
      X = self.dataset.get_X_by_fold(0, kind='test')
      name = name or 'test_preds_fold_0.csv'
      labels = labels or self.dataset.config.labels

    # load dataset table
    df = load_table(self.dataset.config.table_path)
    x_col_name = self.dataset.config.image_column

    print('preds_len:', len(self._preds))
    print('df_len:', len(df))
    print('X_len:', len(X))

    # filter and order the rows of dataset table with predictions
    df = df.set_index(x_col_name).loc[X].reset_index(inplace=False)

    print('df_reindex_len:', len(df))

    # append preds columns
    for index, label in enumerate(labels):
      y_hat = [pred[index] for pred in self._preds]
      print('y_hat_len', len(y_hat))
      df[f'prob_{label}'] = y_hat

    # upload stamps to W&B
    if Experiment.log_wandb:
      with Experiment.Tracker(name=name, job_type='pred'):
        sorted_df = df.sort_values('prob_merger', ascending=False)
        imgs = []
        for i in range(min(50, len(sorted_df))):
          path = sorted_df[x_col_name][i]
          prob = sorted_df['prob_merger'][i]
          label = f'P(M) = {prob*100:.1f}'
          img = wandb.Image(path, caption=label)
          imgs.append(img)
        wandb.log({'predictions': imgs})

    # upload to github
    Experiment.upload_file_gd(name, df)

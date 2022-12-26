from typing import Dict, List

import pandas as pd
import tensorflow as tf

from mergernet.core.experiment import Experiment
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


  def upload(self, name: str = None, label_map: Dict = None):
    # get X by dataset type: predictions or train
    if self.dataset.config.label_column is None:
      X = self.dataset.get_X()
      name = name or 'predictions.csv'
    else:
      X = self.dataset.get_X_by_fold(0, kind='test')
      name = name or 'test_preds_fold_0.csv'
      label_map = label_map or self.dataset.config.labels

    # load dataset table
    df = pd.read_csv(self.dataset.config.table_path)
    x_col_name = self.dataset.config.image_column

    print('preds_len:', len(self._preds))
    print('df_len:', len(df))
    print('X_len:', len(X))

    # filter and order the rows of dataset table with predictions
    df = df.set_index(x_col_name).loc[X].reset_index(inplace=False)

    print('df_reindex_len:', len(df))

    # append preds columns
    for label, index in label_map.items():
      y_hat = [pred[index] for pred in self._preds]
      print('y_hat_len', len(y_hat))
      df[f'prob_{label}'] = y_hat

    # upload to github
    Experiment.upload_file_gd(name, df)

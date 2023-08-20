from typing import List

import numpy as np
import pandas as pd
import pymilvus as pm
from tqdm import tqdm

from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname
from mergernet.data.dataset import Dataset
from mergernet.estimators.decals import ZoobotEstimator


def normalize_cnn_features(features):
  return features / np.linalg.norm(features, axis=1)[:, np.newaxis]


def insert_into_database(
  col,
  iauname: List[str],
  ra: List[float],
  dec: List[float],
  embeddings: List[List[float]]
):
  step = 10_000
  total = int(np.ceil(len(iauname) / step))
  for i in tqdm(range(total), total=total):
    lower = i * step
    upper = (i + 1) * step

    entities = [
      iauname[lower:upper],
      ra[lower:upper],
      dec[lower:upper],
      embeddings[lower:upper]
    ]

    insert_result = col.insert(entities)
    col.flush()


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 12
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Milvus database test
    """
    # Connect Milvus
    pm.connections.connect('default', host='localhost', port='19530')

    fields = [
      pm.FieldSchema(name='iauname', dtype=pm.DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
      pm.FieldSchema(name='ra', dtype=pm.DataType.DOUBLE),
      pm.FieldSchema(name='dec', dtype=pm.DataType.DOUBLE),
      pm.FieldSchema(name='embeddings', dtype=pm.DataType.FLOAT_VECTOR, dim=1280)
    ]

    schema = pm.CollectionSchema(fields, 'First milvus experiment')


    # # Load dataframe of experiment 11
    self.download_file_gd('decals_1M.parquet', 11)
    df = pd.read_parquet(self.local_exp_path / 'decals_1M.parquet')
    print('dataframe size:', len(df))
    df = df.drop_duplicates(subset=['iauname'])
    print('dataframe size after duplication drop:', len(df))


    print('Deleting decals_1M collection')
    pm.utility.drop_collection('decals_1M')
    col = pm.Collection('decals_1M', schema, consistency_level='Strong')
    print('Inserting into decals_1M collection')
    insert_into_database(
      col=col,
      iauname=df.iauname.values,
      ra=df.ra.values,
      dec=df.dec.values,
      embeddings= df[df.columns.values[3:]].values
    )

    print('Deleting decals_1M_normalized collection')
    pm.utility.drop_collection('decals_1M_normalized')
    col = pm.Collection('decals_1M_normalized', schema, consistency_level='Strong')
    print('Inserting into decals_1M_normalized collection')
    insert_into_database(
      col=col,
      iauname=df.iauname.values,
      ra=df.ra.values,
      dec=df.dec.values,
      embeddings=normalize_cnn_features(df[df.columns.values[3:]].values)
    )

    self.autoclean()




if __name__ == '__main__':
  Job().run()

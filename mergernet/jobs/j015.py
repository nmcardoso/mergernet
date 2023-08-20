import numpy as np
import pandas as pd
import pymilvus as pm
from tqdm import tqdm

from mergernet.core.experiment import Experiment
from mergernet.core.utils import Timming, iauname, iauname_path
from mergernet.data.dataset import Dataset
from mergernet.data.dataset_config import DatasetConfig
from mergernet.data.image import Crop, ImagePipeline, LegacyRGB, TensorToImage
from mergernet.estimators.decals import ZoobotEstimator
from mergernet.estimators.similarity import MilvusClusterSimilarity
from mergernet.services.legacy import LegacyService


def normalize_cnn_features(features):
  return features / np.linalg.norm(features, axis=1)[:, np.newaxis]



class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 15
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Good mergers dataset
    """
    self.download_file_gd('merger_only_features.parquet', 14)

    df = pd.read_parquet(self.local_exp_path / 'merger_only_features.parquet')

    pm.connections.connect('default', host='localhost', port='19530')

    fields = [
      pm.FieldSchema(name='iauname', dtype=pm.DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
      pm.FieldSchema(name='ra', dtype=pm.DataType.DOUBLE),
      pm.FieldSchema(name='dec', dtype=pm.DataType.DOUBLE),
      pm.FieldSchema(name='embeddings', dtype=pm.DataType.FLOAT_VECTOR, dim=1280)
    ]

    schema = pm.CollectionSchema(fields, 'First milvus experiment')

    collection = pm.Collection('decals_1M_normalized', schema, consistency_level='Strong')

    collection.release()

    # if collection.has_index():
    #   collection.drop_index()

    index_params = {
      'index_type': 'FLAT',
      'metric_type': 'IP',
      'params': {}
    }

    query_params = {
      'metric_type': 'IP',
    }

    # collection.create_index('embeddings', index_params)

    collection.load()

    normalized_features = normalize_cnn_features(df[df.columns.values[3:]].values)

    mv = MilvusClusterSimilarity()

    t = Timming()

    result_df = mv.cluster_search(
      collection,
      normalized_features,
      55,
      params=query_params,
      fields=['iauname', 'ra', 'dec'],
    )

    print('Search time:', t.end())

    self.upload_file_gd('mergers_5k_similarity.parquet', result_df)

    self.autoclean()



if __name__ == '__main__':
  Job().run()

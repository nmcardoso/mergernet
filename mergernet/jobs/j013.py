import numpy as np
import pandas as pd
import pymilvus as pm
from tqdm import tqdm

from mergernet.core.experiment import Experiment

queries = [
  {
    'name': 'FLAT_L2',
    'index_field': 'embeddings',
    'query_params': {
      'metric_type': 'L2'
    },
    'index_params': {
      'index_type': 'FLAT',
      'metric_type': 'L2',
      'params': {}
    },
  },
  {
    'name': 'IVF_FLAT_L2',
    'index_field': 'embeddings',
    'query_params': {
      'metric_type': 'L2',
      'params': {
        'nprobe': 32,
      },
    },
    'index_params': {
      'index_type': 'IVF_FLAT',
      'metric_type': 'L2',
      'params': {
        'nlist': 16
      },
    },
  },
  {
    'name': 'HNSW_L2',
    'index_field': 'embeddings',
    'query_params': {
      'metric_type': 'L2',
      'params': {
        'ef': 32,
      },
    },
    'index_params': {
      'index_type': 'HNSW',
      'metric_type': 'L2',
      'params': {
        'M': 16,
        'efConstruction': 16
      },
    },
  },
]

queries_normalized = [
  {
    'name': 'FLAT_IP',
    'index_field': 'embeddings',
    'query_params': {
      'metric_type': 'IP'
    },
    'index_params': {
      'index_type': 'FLAT',
      'metric_type': 'IP',
      'params': {}
    },
  },
  {
    'name': 'IVF_FLAT_IP',
    'index_field': 'embeddings',
    'query_params': {
      'metric_type': 'IP',
      'params': {
        'nprobe': 32,
      },
    },
    'index_params': {
      'index_type': 'IVF_FLAT',
      'metric_type': 'IP',
      'params': {
        'nlist': 16
      },
    },
  },
  {
    'name': 'HNSW_IP',
    'index_field': 'embeddings',
    'query_params': {
      'metric_type': 'IP',
      'params': {
        'ef': 32,
      },
    },
    'index_params': {
      'index_type': 'HNSW',
      'metric_type': 'IP',
      'params': {
        'M': 16,
        'efConstruction': 16
      },
    },
  },
]


def do_search(col, queries, vectors_to_search):
  print()

  for i, query in enumerate(queries):
    print(f'[{i + 1} of {len(queries)}] {query["name"]}')

    col.release()

    if col.has_index():
      col.drop_index()

    col.create_index('embeddings', query['index_params'])

    col.load()

    result = col.search(
      vectors_to_search,
      query['index_field'],
      query['query_params'],
      limit=5,
      output_fields=['iauname', 'ra', 'dec']
    )

    for hits in result:
      for hit in hits:
        ra = hit.entity.get('ra')
        dec = hit.entity.get('dec')
        print(f'hit: {hit}')
        print(f'https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra:.6f}&dec={dec:.6f}&pixscale=0.364&size=224')
        print()

    print()


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 13
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Milvus database query
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

    iaunames = ['J060218.60-390608.6', 'J060420.62-293427.2']
    vectors_to_search = df[df.iauname.isin(iaunames)][df.columns.values[3:]].values
    print('vectors_to_search shape', vectors_to_search.shape)


    col = pm.Collection('decals_1M', schema, consistency_level='Strong')
    do_search(col, queries, vectors_to_search)

    col = pm.Collection('decals_1M_normalized', schema, consistency_level='Strong')
    do_search(col, queries_normalized, vectors_to_search)

    self.autoclean()



if __name__ == '__main__':
  Job().run()

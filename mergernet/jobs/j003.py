import pandas as pd

from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname
from mergernet.estimators.similarity import SimilarityEstimator


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 3
    self.log_wandb = False
    self.restart = False


  def call(self):
    Experiment.download_file_gd('dr5_8_b0_pca10_and_safe_ids.parquet', shared=True)
    Experiment.download_file_gd('dr5_dr8_catalog_with_radius.parquet', shared=True)

    df_pca = pd.read_parquet(Experiment.local_exp_path / 'dr5_8_b0_pca10_and_safe_ids.parquet')
    df_meta = pd.read_parquet(Experiment.local_exp_path / 'dr5_dr8_catalog_with_radius.parquet')

    df = pd.merge(
      df_meta[['ra', 'dec', 'galaxy_id']],
      df_pca,
      how='inner',
      on='galaxy_id'
    ).drop('galaxy_id', axis=1)

    df['iauname'] = iauname(df['ra'].values, df['dec'].values)

    Experiment.upload_file_gd('decals_pca10.csv', df)


if __name__ == '__main__':
  Job().run()

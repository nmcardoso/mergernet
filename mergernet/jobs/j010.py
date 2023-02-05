import pandas as pd

from mergernet.core.experiment import Experiment
from mergernet.data.dataset import Dataset
from mergernet.estimators.decals import ZoobotEstimator


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 10
    self.log_wandb = False
    self.restart = False

  def call(self):
    names = [f'decals_cnn_representations_part{i}.parquet' for i in range(13)]

    for name in names:
      self.download_file_gd(name, 9)

    dfs = [pd.read_parquet(self.local_exp_path / name) for name in names]

    df = pd.concat(dfs)

    feat_cols = [col for col in df.columns.values if col.startswith('feat_')]
    features = df[feat_cols].to_numpy()

    model = ZoobotEstimator(
      hp=None,
      dataset=None,
      config=None,
      crop_size=224,
      resize_size=224,
    )

    pca_df = model.pca(features, 10, include_iauname=False)
    pca_df.insert(0, 'iauname', df['iauname'].values)
    self.upload_file_gd('decals_cnn_pca_10.parquet', pca_df)

    pca_df = model.pca(features, 20, include_iauname=False)
    pca_df.insert(0, 'iauname', df['iauname'].values)
    self.upload_file_gd('decals_cnn_pca_20.parquet', pca_df)

    pca_df = model.pca(features, 30, include_iauname=False)
    pca_df.insert(0, 'iauname', df['iauname'].values)
    self.upload_file_gd('decals_cnn_pca_30.parquet', pca_df)

    pca_df = model.pca(features, 40, include_iauname=False)
    pca_df.insert(0, 'iauname', df['iauname'].values)
    self.upload_file_gd('decals_cnn_pca_40.parquet', pca_df)

    pca_df = model.pca(features, 50, include_iauname=False)
    pca_df.insert(0, 'iauname', df['iauname'].values)
    self.upload_file_gd('decals_cnn_pca_50.parquet', pca_df)

    pca_df = model.pca(features, 100, include_iauname=False)
    pca_df.insert(0, 'iauname', df['iauname'].values)
    self.upload_file_gd('decals_cnn_pca_100.parquet', pca_df)


if __name__ == '__main__':
  Job().run()

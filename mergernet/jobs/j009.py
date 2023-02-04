import pandas as pd

from mergernet.core.experiment import Experiment
from mergernet.data.dataset import Dataset
from mergernet.estimators.decals import ZoobotEstimator


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 9
    self.log_wandb = False
    self.restart = False

  def call(self):
    dataset_configs = [
      Dataset.registry.DECALS_0364_1M_PART0,
      Dataset.registry.DECALS_0364_1M_PART1,
      Dataset.registry.DECALS_0364_1M_PART2,
      Dataset.registry.DECALS_0364_1M_PART3,
      Dataset.registry.DECALS_0364_1M_PART4,
      Dataset.registry.DECALS_0364_1M_PART5,
      Dataset.registry.DECALS_0364_1M_PART6,
      Dataset.registry.DECALS_0364_1M_PART7,
      Dataset.registry.DECALS_0364_1M_PART8,
      Dataset.registry.DECALS_0364_1M_PART9,
      Dataset.registry.DECALS_0364_1M_PART10,
      Dataset.registry.DECALS_0364_1M_PART11,
      Dataset.registry.DECALS_0364_1M_PART12,
    ]

    for i, ds_config in enumerate(dataset_configs):
      ds = Dataset(ds_config)

      model = ZoobotEstimator(
        hp=None,
        dataset=ds,
        config=ZoobotEstimator.registry.ZOOBOT_GREYSCALE,
        crop_size=224,
        resize_size=224,
      )

      table_name = f'decals_cnn_representations_part{i}.parquet'
      pca_table_name = f'decals_cnn_pca_part{i}.parquet'

      model.cnn_representations(table_name)
      self.upload_file_gd(table_name)

      if i == 0:
        model.plot('model.png')
        self.upload_file_gd('model.png')

      df = pd.read_parquet(self.local_exp_path / table_name)
      feat_cols = [col for col in df.columns.values if col.startswith('feat_')]
      features = df[feat_cols].to_numpy()

      model.pca(features, 10, pca_table_name)
      self.upload_file_gd(pca_table_name)




if __name__ == '__main__':
  Job().run()

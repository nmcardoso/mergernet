import pandas as pd

from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname
from mergernet.data.dataset import Dataset
from mergernet.estimators.decals import ZoobotEstimator


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 11
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Create Decals 1M dataset metadata table (index, iauname, ra, dec) to use
    in faiss similarity search
    """
    # Catalog dataframe
    self.download_file_gd('dr5_dr8_catalog_with_radius.parquet', shared=True)
    cat_df = pd.read_parquet(self.local_exp_path / 'dr5_dr8_catalog_with_radius.parquet')
    print('Catalog dataframe columns:', ', '.join(cat_df.columns.values))
    print('Computing iaunames')
    cat_df['iauname'] = iauname(cat_df['ra'], cat_df['dec'])
    cat_df = cat_df[['iauname', 'ra', 'dec']]

    # Representations dataframe
    names = [f'decals_cnn_representations_part{i}.parquet' for i in range(13)]
    paths = [self.local_exp_path / name for name in names]

    df = None

    for i, (name, path) in enumerate(zip(names, paths)):
      print(f'[{i + 1} of {len(names)}] Inner join')
      if not path.exists():
        self.download_file_gd(name, 9)

      rep_df = pd.read_parquet(path)
      print('rows before join:', len(rep_df))
      rep_df = rep_df.merge(cat_df, on='iauname')
      rep_df.insert(1, 'ra', rep_df.pop('ra'))
      rep_df.insert(2, 'dec', rep_df.pop('dec'))
      print('rows after join:', len(rep_df))
      print()

      if i == 0:
        df = rep_df
      else:
        df = pd.concat([df, rep_df])

    self.upload_file_gd('decals_1M_positions_meta.parquet', df)

    self.autoclean()




if __name__ == '__main__':
  Job().run()

import faiss
import pandas as pd

from mergernet.core.experiment import Experiment
from mergernet.data.dataset import Dataset
from mergernet.estimators.decals import ZoobotEstimator


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 17
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Test FAISS indexes
    """
    # paths = [
    #   self.local_exp_path / f'decals_cnn_representations_part{i}.parquet'
    #   for i in range(13)
    # ]

    # for path in paths:
    #   if not path.exists():
    #     self.download_file_gd(path, 9)

    # dfs = [pd.read_parquet(path) for path in paths]
    # df = pd.concat(dfs)


    fname = 'decals_cnn_representations_part0.parquet'
    path = self.local_exp_path / fname

    self.download_file_gd(fname, 9)

    df = pd.read_parquet(path)
    features = df[df.columns[1:]].values
    dim = features.shape[1]
    query = features[1212]
    neighbors = 10

    idxL2 = faiss.IndexFlatL2(dim)
    print('Index L2 trained:', idxL2.is_trained)
    idxL2.add(features)
    idxL2.search(query, neighbors)




    self.delete(path)


if __name__ == '__main__':
  Job().run()

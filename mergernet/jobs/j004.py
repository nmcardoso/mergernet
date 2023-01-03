import pandas as pd
from tqdm import tqdm

from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname, iauname_relative_path
from mergernet.estimators.similarity import SimilarityEstimator
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 4
    self.log_wandb = False
    self.restart = False


  def call(self):
    Experiment.download_file_gd('decals_pca10.csv', exp_id=3)
    Experiment.download_file_gd('representations_pca.csv', exp_id=2)

    reference_table = pd.read_csv(Experiment.local_exp_path / 'decals_pca10.csv')
    additional_table = pd.read_csv(Experiment.local_exp_path / 'representations_pca.csv')

    sim = SimilarityEstimator([reference_table, additional_table])

    (Experiment.local_exp_path / 'preds').mkdir(parents=True, exist_ok=True)
    ls = LegacyService(workers=5)

    for row in tqdm(additional_table.itertuples()):
      query_gal, neighbours = sim.predict(ra=row.ra, dec=row.dec)
      neighbours.to_csv(
        Experiment.local_exp_path / 'preds' / f'{row.iauname}.csv',
        index=False
      )

      n = neighbours.sort_values(by='knn_distance', axis=0, ascending=True)[:40]

      paths = iauname_relative_path(
        iauname(n.ra.values, n.dec.values),
        Experiment.local_exp_path / 'img',
        '.jpg'
      )

      ls.cutout(
        query_gal.ra,
        query_gal.dec,
        iauname_relative_path(
          iauname(query_gal.ra, query_gal.dec),
          Experiment.local_exp_path / 'img',
          '.jpg'
        )
      )

      ls.batch_cutout(
        n.ra.values,
        n.dec.values,
        paths
      )

      SimilarityEstimator.plot(
        Experiment.local_exp_path / f'plot_{row.iauname}.pdf',
        query_gal,
        n,
        Experiment.local_exp_path / 'img',
        '.jpg'
      )


if __name__ == '__main__':
  Job().run()

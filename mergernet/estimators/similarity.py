import logging
from pathlib import Path
from typing import Iterable, Sequence, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors

from mergernet.core.utils import Timming, iauname_relative_path, load_image
from mergernet.estimators.base import Estimator

L = logging.getLogger(__name__)


class SimilarityEstimator(Estimator):
  """
  Similarity Estimator

  Parameters
  ----------
  pca_table: str
    This table must ``ra``, ``dec``, ``feat_0_pca``, ``feat_1_pca``, ...,
    ``feat_n_pca`` columns.
  """
  def __init__(self, pca_tables: Iterable[pd.DataFrame]):
    super().__init__(None, None)

    self.pca_table = pd.concat(
      [t.reset_index(drop=True) for t in pca_tables],
      axis=0
    )

    self.coords = SkyCoord(
      self.pca_table['ra'],
      self.pca_table['dec'],
      unit='deg'
    )

    self.features = self.pca_table[[
      col for col in self.pca_table.columns.values
      if col.endswith('_pca')
    ]].values


  def _match(self, ra: float, dec: float):
    L.info(f'Crossmatching object ({ra:.4}, {dec:.4}) against {len(self.coords)} objects')
    t = Timming()
    search_coord = SkyCoord(ra, dec, unit='deg')
    best_index, separation, _ = match_coordinates_sky(search_coord, self.coords)
    L.info(f'Crossmatch done in {t.end()}')
    return best_index, separation


  def _get_nontrivial_neighbours(
    self,
    query_galaxy: pd.Series,
    neighbours: pd.DataFrame,
    min_sep=100*u.arcsec
  ):
    query_coord = SkyCoord(query_galaxy['ra'], query_galaxy['dec'], unit='deg')
    separations = [
      SkyCoord(other['ra'], other['dec'], unit='deg').separation(query_coord)
      for _, other in neighbours.iterrows()
    ]
    above_min_sep = np.array([sep > min_sep for sep in separations])

    L.info(f'Removed {(~above_min_sep).sum()} sources within {min_sep} arcsec of target galaxy')

    return neighbours[above_min_sep].reset_index(drop=True)


  def build(self, n_neighbors=500, metric='manhattan'):
    L.info('Start of model build')
    t = Timming()
    self._tf_model = NearestNeighbors(
      n_neighbors=n_neighbors,
      algorithm='ball_tree',
      metric=metric
    )
    L.info(f'End of model build. Elapsed time: {t.end()}')


  def train(self):
    L.info('Start of model fit')
    t = Timming()
    self.neighbors = self._tf_model.fit(self.features)
    L.info(f'End of model fit. Elapsed time: {t.end()}')


  def predict(
    self,
    ra: float,
    dec: float,
    max_separation: float = 2
  ):
    if self._tf_model is None:
      self.build()
      self.train()

    best_index, separation = self._match(ra, dec)

    if separation > max_separation*u.arcsec:
      raise ValueError(f'Object not found. Separation: {separation} arcsec')

    distances, indices = self.tf_model.kneighbors(self.features[best_index].reshape(1, -1))
    neighbour_indices = np.squeeze(indices)

    assert neighbour_indices[0] == best_index  # should find itself

    query_galaxy = self.pca_table.iloc[best_index].copy(deep=True)
    query_galaxy['knn_distance'] = distances[0][0]
    neighbours = self.pca_table.iloc[neighbour_indices[1:]].copy(deep=True)
    neighbours['knn_distance'] = distances[0][1:]
    neighbours = neighbours[['iauname', 'ra', 'dec', 'knn_distance']]

    nontrivial_neighbours = self._get_nontrivial_neighbours(query_galaxy, neighbours)

    return query_galaxy, nontrivial_neighbours


  @staticmethod
  def plot(
    save_path: Union[str, Path],
    query_df: pd.DataFrame,
    neighbours_df: pd.DataFrame,
    images_path: Union[str, Path],
    image_type: str = '.jpg'
  ):
    neighbours_df = neighbours_df.reset_index(drop=True)

    with PdfPages(save_path) as pdf:
      fig = plt.figure(figsize=(8.3, 11.7))
      plt.imshow(
        load_image(
          iauname_relative_path(query_df.iauname, images_path, image_type)
        )
      )
      plt.gca().set_xticks([])
      plt.gca().set_yticks([])
      plt.title('Query Galaxy')
      pdf.savefig(fig)
      plt.close()

      pages = int(np.ceil(len(neighbours_df) / 20))

      for page in range(pages):
        fig, axs = plt.subplots(
          ncols=4,
          nrows=5,
          figsize=(8.3, 11.7),
          layout='constrained'
        )
        for row in range(5):
          for col in range(4):
            neighbour = neighbours_df.iloc[20 * page + 4 * row + col]
            axs[row, col].imshow(
              load_image(
                iauname_relative_path(neighbour.iauname, images_path, image_type)
              )
            )
            axs[row, col].title(f'{neighbour.ra:.3f}, {neighbour.dec:.3f} | ({neighbour.knn_distance:.2f})')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        pdf.savefig(fig)
        plt.close()



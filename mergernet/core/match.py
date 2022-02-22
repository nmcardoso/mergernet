from pathlib import Path
from typing import Sequence, Tuple, Union

import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky



arcsec = 1. / 3600



class XTable:
  def __init__(
    self,
    ra: str,
    dec: str,
    df: pd.DataFrame = None,
    path: str = None,
    columns: Sequence[str] = None
  ):
    self.path = path
    self.df = df
    self.ra = ra
    self.dec = dec
    self.columns = columns
    self.coords = None

  def get_coords(self):
    if self.coords is None:
      df = self.to_df()
      ra = df[self.ra].to_list()
      dec = df[self.dec].to_list()
      self.coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

      if self.path is not None:
        del df
    return self.coords

  def to_df(self):
    if self.path is not None:
      df = pd.read_csv(self.path)
    else:
      df = self.df
    return df



class CrossMatchResult:
  table: pd.DataFrame = None
  primary_idx: np.ndarray = None
  secondary_idx: np.ndarray = None
  distance: np.ndarray = None




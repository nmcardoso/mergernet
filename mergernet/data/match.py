from pathlib import Path
from typing import Sequence, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky

from mergernet.core.utils import load_table

arcsec_in_deg = 1. / 3600



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
      df = load_table(self.path)
    else:
      df = self.df
    return df



class CrossMatchResult:
  table: pd.DataFrame = None
  primary_idx: np.ndarray = None
  secondary_idx: np.ndarray = None
  distance: np.ndarray = None



class CrossMatch:
  def pair_match(
    self,
    table1: XTable,
    table2: XTable,
    nthneighbor: int = 1
  ) -> Tuple[np.ndarray, np.ndarray]:
    df1_coord = table1.get_coords()
    df2_coord = table2.get_coords()

    idx, d, _ = match_coordinates_sky(
      df1_coord,
      df2_coord,
      nthneighbor=nthneighbor
    )

    return np.array(idx), np.array(d)


  def crossmatch(
    self,
    table1: XTable,
    table2: XTable,
    radius: float = 1
  ):
    idx, d = self.pair_match(table1, table2)

    df1 = table1.to_df()
    df2 = table2.to_df()

    mask = d < (radius * arcsec_in_deg)

    primary_idx = mask.nonzero()[0]
    secondary_idx = idx[mask]

    df1_masked = df1.iloc[primary_idx]
    df2_masked = df2.iloc[secondary_idx]

    df1_subsample = df1_masked[table1.columns].copy() \
      if table1.columns is not None else df1_masked.copy()
    df2_subsample = df2_masked[table2.columns].copy() \
      if table2.columns is not None else df2_masked.copy()

    for col in df2_subsample.columns.tolist():
      df1_subsample[col] = df2_subsample[col].to_numpy()
      # TODO: include a flag "replace" in this method to indicate if t2 must
      # replace or not t1 columns. This implementation consider replace=True.

    r = CrossMatchResult()
    r.distance = d[mask]
    r.primary_idx = primary_idx
    r.secondary_idx = secondary_idx
    r.table = df1_subsample
    return r


  def unique(
    self,
    table: XTable,
    radius: float
  ):
    idx, d = self.pair_match(table, table, nthneighbor=2)

    mask = d < (radius * arcsec_in_deg)
    primary_idx = mask.nonzero()[0]
    secondary_idx = idx[mask]
    removed_idx = []

    for pid, sid in zip(primary_idx, secondary_idx):
      if sid not in removed_idx:
        removed_idx.append(pid)

    del_mask = np.isin(idx, removed_idx, invert=True).nonzero()[0]

    df = table.to_df()

    r = CrossMatchResult()
    r.distance = d[del_mask]
    r.table = df.iloc[del_mask]
    r.primary_idx = primary_idx
    r.secondary_idx = secondary_idx
    return r



if __name__ == '__main__':
  cm = CrossMatch()

  t1 = XTable(
    ra='ra',
    dec='dec',
    path=Path(__file__ + '/../../../data/zoo2Stripe82Coadd1.csv.gz').resolve(),
  )

  r = cm.unique(t1, 30)
  print(r.table)
  print(len(t1.to_df()), len(r.table))

import argparse
from pathlib import Path
import sys
import os
from typing import Union
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.core.utils import Timming, heading, CachedDataFrame
from mergernet.core.constants import DATA_ROOT, RANDOM_SEED
from mergernet.data.match import XTable, CrossMatch
from mergernet.services.sdss import SloanService
from mergernet.data.dataset import Dataset
from mergernet.visualization.plot import mag_class_distribution, conf_matrix, object_position

arcsec_in_deg = 1. / 3600

# Catalogs
gz1_path = DATA_ROOT / 'catalogs/zoo1_spec_all.csv.gz'
gzd_5_path = DATA_ROOT / 'catalogs/gz_decals_volunteers_5_mag.csv.gz'
gzd_auto_path = DATA_ROOT / 'catalogs/gz_decals_auto_posteriors.csv.gz'
gzd_12_path = DATA_ROOT / 'catalogs/gz_decals_volunteers_1_and_2_mag.csv.gz'
darg_path = DATA_ROOT / 'catalogs/darg_mergers.csv.gz'

# Visual Inspections
merger_exclude_path = DATA_ROOT / 'inspection/merger_exclude.csv'
merger_include_path = DATA_ROOT / 'inspection/merger_include.csv'
disturbed_exclude_path = DATA_ROOT / 'inspection/disturbed_exclude.csv'
disturbed_include_path = DATA_ROOT / 'inspection/disturbed_include.csv'

# Values for self-match
pixel_scale = 0.55
stamp_size = 128
exclusion_radius = pixel_scale * stamp_size * np.sqrt(2) / 2
exclusion_radius *= 1.15 # add 15% offset

# Final dataset kfold params
kfold_splits = 5
kfold_bins = 6



def prepare_gzd(df: pd.DataFrame,  class_name: str, sample_name: str):
  df = df[['iauname', 'ra', 'dec', 'redshift', 'mag_r']]
  df = df.rename(columns={'redshift': 'z'})
  df['class'] = [class_name] * len(df)
  df['sample'] = [sample_name] * len(df)
  return df


def prepare_darg(df: pd.DataFrame, class_name: str):
  df = df[['iauname', 'ra1', 'dec1', 'specz1', 'r_app_1']]
  df = df.rename(columns={
    'ra1': 'ra',
    'dec1': 'dec',
    'specz1': 'z',
    'r_app_1': 'mag_r'
  })
  df['class'] = [class_name] * len(df)
  df['sample'] = ['darg'] * len(df)
  return df



def prepare_gz1(df: pd.DataFrame, class_name: str):
  df = df[['iauname', 'ra', 'dec', 'z', 'modelMag_r']]
  df = df.rename(columns={'modelMag_r': 'mag_r'})
  df['class'] = [class_name] * len(df)
  df['sample'] = ['gz1'] * len(df)
  return df



def create_merger_dataset():
  gzd_5 = CachedDataFrame.load(gzd_5_path)
  gzd_12 = CachedDataFrame.load(gzd_12_path)
  gzd_auto = CachedDataFrame.load(gzd_auto_path)
  darg = CachedDataFrame.load(darg_path)
  merger_include = CachedDataFrame.load(merger_include_path)
  merger_exclude = CachedDataFrame.load(merger_exclude_path)

  # GalazyZoo Decals 5
  merger_gzd5_mask = (
    (gzd_5['merging_merger_fraction'] > 0.7)
    | gzd_5['iauname'].isin(merger_include['iauname'])
  ) & (gzd_5['mag_r'].between(12, 20))
  merger_gzd5 = prepare_gzd(gzd_5[merger_gzd5_mask], 'merger', 'gzd_5')

  # GalazyZoo Decals 1, 2
  merger_gzd12_mask = (
    (gzd_12['merging_merger_fraction'] > 0.7)
    | gzd_12['iauname'].isin(merger_include['iauname'])
  ) & (gzd_12['mag_r'].between(12, 20))
  merger_gzd12 = prepare_gzd(gzd_12[merger_gzd12_mask], 'merger', 'gzd_12')

  # GalaxyZoo Decals Auto
  merger_gzd_auto_mask = (
    (gzd_auto['merging_merger_fraction'] > 0.7)
    | gzd_auto['iauname'].isin(merger_include['iauname'])
  ) & (gzd_auto['mag_r'].between(12, 20))
  merger_gzdauto = prepare_gzd(gzd_auto[merger_gzd_auto_mask], 'merger', 'gzd_auto')

  # Darg
  merger_darg_mask = (
    (darg['stage'] == 2) | (darg['stage'] == 3)
  ) & darg['r_app_1'].between(12, 20)
  merger_darg = prepare_darg(darg[merger_darg_mask], 'merger')

  # Dataset concatenation
  merger = pd.concat(
    [merger_gzdauto, merger_gzd5, merger_gzd12, merger_darg],
    ignore_index=True
  ).copy()
  merger = merger.drop_duplicates(subset=['iauname'])
  merger = merger[(~merger['iauname'].isin(merger_exclude))]

  # Remove multiple table objects in same stamp fov using self-match
  cm = CrossMatch()
  merger = cm.unique(XTable('ra', 'dec', merger), exclusion_radius).table

  print()
  heading('Merger Class')
  print('Merger GZ Decals 5:', len(merger_gzd5), '->', len(merger[merger['sample'] == 'gzd_5']))
  print('Merger GZ Decals 1,2:', len(merger_gzd12), '->', len(merger[merger['sample'] == 'gzd_12']))
  print('Merger GZ Decals Auto:', len(merger_gzdauto), '->', len(merger[merger['sample'] == 'gzd_auto']))
  print('Merger Darg:', len(merger_darg), '->', len(merger[merger['sample'] == 'darg']))
  print('Merger (final):', len(merger))
  print('Merger Columns:', ', '.join(merger.columns), f'({len(merger.columns)})')
  return merger




def create_non_merger_dataset():
  gz1 = CachedDataFrame.load(gz1_path)

  non_merger_mask = (
    (gz1['p_cw'] == 1)
    | (gz1['p_acw'] == 1)
    | (gz1['p_el'] == 1)
  ) & (gz1['uncertain'] == 0) & (gz1['modelMag_r'].between(12, 20))
  non_merger_gz1 = prepare_gz1(gz1[non_merger_mask], 'non_merger')

  non_merger = non_merger_gz1

  # Remove multiple table objects in same stamp fov using self-match
  cm = CrossMatch()
  non_merger = cm.unique(XTable('ra', 'dec', non_merger), exclusion_radius).table

  if len(non_merger) > 6500:
    non_merger = non_merger.sample(n=6500, random_state=RANDOM_SEED)

  print()
  heading('Non Merger Class')
  print('Non Merger (final):', len(non_merger))
  print('Non Merger Columns:', ', '.join(non_merger.columns), f'({len(non_merger.columns)})')
  return non_merger




def bin_ds(save: bool = False):
  if not save:
    print('\nRunning in DRY-RUN mode')

  mergers = create_merger_dataset()
  non_mergers = create_non_merger_dataset()

  # check for overlaps between classes and exclude non_mergers
  cm = CrossMatch()
  overlaps = cm.crossmatch(
    XTable('ra', 'dec', mergers),
    XTable('ra', 'dec', non_mergers),
    radius=exclusion_radius
  )
  exclude_iauname = non_mergers['iauname'].iloc[overlaps.secondary_idx]
  non_mergers = non_mergers[~non_mergers['iauname'].isin(exclude_iauname)]

  print()
  heading('Class Join')
  print('Dataset size per class after x-match between classes')
  print('Merger:', len(mergers))
  print('Non Merger:', len(non_mergers))

  # Concatenate all classes
  bin_ds = pd.concat([mergers, non_mergers], ignore_index=True).copy()
  bin_ds = bin_ds.drop_duplicates(subset='iauname')

  # Concat fold column for cross validation
  bin_ds = Dataset.concat_fold_column(
    df=bin_ds,
    fname_column='iauname',
    class_column='class',
    r_column='mag_r',
    n_splits=kfold_splits,
    bins=kfold_bins
  )
  print()
  heading('Final Dataset')
  print('Number of splits:', kfold_splits)
  print('Number of bins:', kfold_bins)
  print('Columns:', ', '.join(bin_ds.columns), f'({len(bin_ds.columns)})')

  save_path = DATA_ROOT / 'prepared' / 'bin_ds.csv'
  save_path.parent.mkdir(parents=True, exist_ok=True)
  if save:
    bin_ds.to_csv(save_path, index=False)
    print(f'Table saved in {str(save_path)}')




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-s',
    action='store_true',
    help='[Optional] Save the generated table.',
    default=False
  )
  args = parser.parse_args()

  t = Timming()
  bin_ds(save=args.s)
  print(f'\nEllapsed Time: {t.end()}')

from pathlib import Path
import sys
import os
import shutil
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.services.legacy import LegacyService


def download(df, img_folder, ra='ra', dec='dec'):
  img_folder = Path(img_folder)
  if not img_folder.exists():
    img_folder.mkdir(parents=True, exist_ok=True)

  s = LegacyService()

  paths = [img_folder / f'{name}.jpg' for name in df['iauname'].tolist()]

  s.batch_download_rgb(
    ra=df[ra].to_numpy(),
    dec=df[dec].to_numpy(),
    save_path=paths,
    workers=4,
    replace=False,
    width=128,
    height=128,
    pixscale=0.55,
    use_dev=True,
    layer='ls-dr10-early-grz'
  )

# gz2
# main_photo = pd.read_csv('../data/zoo2MainPhotoz.csv.gz', compression='gzip')
# main = pd.read_csv('../data/zoo2MainSpecz.csv.gz', compression='gzip')
# s82 = pd.read_csv('../data/zoo2Stripe82Normal.csv.gz', compression='gzip')
# df = pd.concat([main, s82, main_photo])

# darg
# df = pd.read_csv('../data/catalogs/darg_mergers.csv.gz')

# gz decals
# gzauto = pd.read_csv('../data/catalogs/gz_decals_volunteers_5.csv.gz')
# df = gzauto[gzauto['spiral-arm-count_1_debiased'] > 0.6]

# mesd
df = pd.read_csv('../data/blind_splus_gal80_r13.5-17.csv')

download(df, '../data/images/blind_splus_gal80_r13.5-17_ls10_rgb_128', 'RA', 'DEC')

from pathlib import Path
import sys
import os
import shutil
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.services.sdss import SloanService


def download(df, img_folder, ra='ra', dec='dec'):
  img_folder = Path(img_folder)
  if not img_folder.exists():
    img_folder.mkdir(parents=True, exist_ok=True)

  s = SloanService()

  paths = [img_folder / f'{name}.jpg' for name in df['iauname'].tolist()]

  s.batch_download_rgb(
    ra=df[ra].to_numpy(),
    dec=df[dec].to_numpy(),
    save_path=paths,
    workers=10,
    replace=False,
    width=128,
    height=128,
    scale=0.55
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
# df = pd.read_csv('../data/mesd.csv')

# bin
df = pd.read_csv('../data/bin_ds.csv')

download(df, '../data/images/bin_sdss_128', 'ra', 'dec')

from pathlib import Path
import sys
import os
import shutil

project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.core.match import XTable, CrossMatch
from mergernet.services.sdss import SloanService


def download(df, img_folder, ra='ra', dec='dec'):
  s = SloanService()

  paths, flag = s.get_image_filename(
    dr7objid=df.dr7objid.to_numpy(),
    dr8objid=df.dr8objid.to_numpy(),
    extension='.jpg',
    basepath=img_folder
  )

  s.batch_download_rgb(
    ra=df[ra].to_numpy(),
    dec=df[dec].to_numpy(),
    save_path=paths,
    workers=8,
    replace=False,
    width=128,
    height=128,
    scale=0.55
  )



t = XTable('../data/zoo2Stripe82Coadd1.csv.gz', 'ra', 'dec')
cm = CrossMatch()
r = cm.unique(t, 30)

df = t.to_df().iloc[r.primary_idx]
ra = df['ra'].to_numpy()
dec = df['dec'].to_numpy()
paths = [f'../data/test/{_id}.jpg' for _id in df['stripe82objid']]

s = SloanService()
s.batch_download_rgb(
  ra=ra,
  dec=dec,
  save_path=paths,
  workers=8,
  replace=True,
  width=128,
  height=128,
  scale=0.55
)

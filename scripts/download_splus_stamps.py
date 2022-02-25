from pathlib import Path
import sys
import os
import shutil
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.services.splus import SplusService, ImageType


df = pd.read_csv('../data/splus_south_gal80.csv')


ra = df['RA'].to_list()
dec = df['DEC'].to_list()
save_path_1 = [f'../data/splus_trilogy_128/{_id}.png' for _id in df['ID']]
save_path_2 = [f'../data/splus_lupton_150/{_id}.png' for _id in df['ID']]
save_path_3 = [f'../data/splus_trilogy_150/{_id}.png' for _id in df['ID']]

splus = SplusService()
splus.batch_image_download(
  ra=ra,
  dec=dec,
  save_path=save_path_1,
  img_type=ImageType.trilogy,
  replace=False,
  size=128,
  workers=3
)

splus.batch_image_download(
  ra=ra,
  dec=dec,
  save_path=save_path_2,
  img_type=ImageType.lupton,
  replace=False,
  size=150,
  workers=5
)

splus.batch_image_download(
  ra=ra,
  dec=dec,
  save_path=save_path_3,
  img_type=ImageType.trilogy,
  replace=False,
  size=150,
  workers=3
)

from pathlib import Path
import sys
import os
import shutil
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.services.splus import SplusService, ImageType
from mergernet.core.constants import SPLUS_BANDS


df = pd.read_csv('../data/blind_splus_gal80_r17_mflag0.csv')


ra = df['RA'].to_list()
dec = df['DEC'].to_list()
save_paths = {
  band: [
    f'../data/images/blind_splus_fits_128/{band}/{_id}.fits'
    for _id in df['ID']
  ] for band in SPLUS_BANDS
}
# save_path_1 = [f'../data/images/blind_splus_trilogy_128/{_id}.png' for _id in df['ID']]
# save_path_2 = [f'../data/images/blind_splus_lupton_150/{_id}.png' for _id in df['ID']]
# save_path_3 = [f'../data/images/blind_splus_trilogy_150/{_id}.png' for _id in df['ID']]

splus = SplusService()
for i, band in enumerate(SPLUS_BANDS):
  print(f'[{i+1}/{len(SPLUS_BANDS)}] Downloading band: {band}')
  splus.batch_image_download(
    ra=ra,
    dec=dec,
    save_path=save_paths[band],
    img_type=ImageType.fits,
    replace=False,
    size=128,
    workers=18,
    band=band
  )

# splus.batch_image_download(
#   ra=ra,
#   dec=dec,
#   save_path=save_path_2,
#   img_type=ImageType.lupton,
#   replace=False,
#   size=150,
#   workers=5
# )

# splus.batch_image_download(
#   ra=ra,
#   dec=dec,
#   save_path=save_path_3,
#   img_type=ImageType.trilogy,
#   replace=False,
#   size=150,
#   workers=8
# )

from pathlib import Path
import sys
import os
import shutil
from tkinter import Image
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
from mergernet.services.splus import SplusService, ImageType


df = pd.read_csv('../data/stripe-0023.csv')


ra = df['RA'].to_list()
dec = df['DEC'].to_list()
save_path = [f'../data/splus_lupton_128/{_id}.png' for _id in df['ID']]

splus = SplusService()
splus.batch_image_download(
  ra=ra,
  dec=dec,
  save_path=save_path,
  img_type=ImageType.lupton,
  replace=False,
  size=128,
  workers=3
)

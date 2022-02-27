from pathlib import Path
import sys
import os
import shutil
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
from mergernet.services.sdss import SloanService



df = pd.read_csv(Path(__file__ + '/../../data/zoo2Stripe82Coadd1.csv.gz').resolve())
ra = df.ra.tolist()
dec = df.dec.tolist()


s = SloanService()
tb = s.crossmatch(
  ra=ra,
  dec=dec,
  fields={
    'SpecObj': {
      'columns': ['z'],
      'join': 'SpecObj.specObjID = x.specObjID'
    },
    'PhotoTag': {
      'columns': ['ra', 'dec', 'modelMag_r'],
      'join': 'PhotoTag.objID = SpecObj.bestObjID'
    }
  },
  search_type='spectro',
  save_path='xmatch.csv',
  workers=5,
  chunk_size=100
)

from pathlib import Path
import sys
import os
import shutil

project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

import pandas as pd
import numpy as np
import mergernet.services.sdss
import mergernet.core.dataset
import mergernet.core.utils


IMG_FOLDER = Path('../data/sdss_lupton_rgb')


zoo1 = pd.read_csv('../data/zoo1_spec_all.csv')
darg = pd.read_csv('../data/tables/darg_mergers.csv')




# spiral = zoo1[(zoo1.p_cs_debiased == 1) & (zoo1.uncertain == 0)]
# ellip = zoo1[(zoo1.p_el_debiased > 0.95) & (zoo1.uncertain == 0)]
# merger = zoo1[(zoo1.p_mg > 0.4)]



s = mergernet.services.sdss.SloanService()

dmerger =  zoo1[
  (zoo1.dr7objid.isin(darg.object1) | zoo1.dr7objid.isin(darg.object2)) &
  (zoo1.modelMag_r < 17.5) &
  (zoo1.modelMag_r > 17) &
  ~zoo1.dr7objid.isin([int(_x.name[4:-4]) for _x in Path('../data/sdss_lupton_jpg_128').iterdir()])
]


# spiral_paths, flag = s.get_image_filename(
#   dr7objid=spiral.dr7objid.to_numpy(),
#   dr8objid=spiral.dr8objid.to_numpy(),
#   extension='.jpg',
#   basepath=IMG_FOLDER
# )

# ellip_paths, flag = s.get_image_filename(
#   dr7objid=ellip.dr7objid.to_numpy(),
#   dr8objid=ellip.dr8objid.to_numpy(),
#   extension='.jpg',
#   basepath=IMG_FOLDER
# )

dmerger_paths, flag = s.get_image_filename(
  dr7objid=dmerger.dr7objid.to_numpy(),
  dr8objid=dmerger.dr8objid.to_numpy(),
  extension='.jpg',
  basepath=IMG_FOLDER
)



# s.batch_download_rgb(
#   ra=spiral.ra.to_numpy(),
#   dec=spiral.dec.to_numpy(),
#   save_path=spiral_paths,
#   workers=2,
#   replace=False,
#   width=128,
#   height=128,
#   scale=0.55
# )

# s.batch_download_rgb(
#   ra=ellip.ra.to_numpy(),
#   dec=ellip.dec.to_numpy(),
#   save_path=ellip_paths,
#   workers=2,
#   replace=False,
#   width=128,
#   height=128,
#   scale=0.55
# )

s.batch_download_rgb(
  ra=dmerger.ra.to_numpy(),
  dec=dmerger.dec.to_numpy(),
  save_path=dmerger_paths,
  workers=8,
  replace=False,
  width=128,
  height=128,
  scale=0.55
)

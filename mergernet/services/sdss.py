import chunk
from pathlib import Path
import secrets
from typing import Dict, List, Sequence, Tuple, Union
import concurrent.futures
import urllib.parse

import numpy as np
import tqdm
import requests
import pandas as pd

from mergernet.core.utils import array_fallback, save_table
from mergernet.services.utils import append_query_params, batch_download_file, download_file


SDSS_RGB_URL = 'http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg'
XID_URL = 'http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/CrossIdSearch'



class SloanService:
  def download_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Path,
    width: int = 256,
    height: int = 256,
    scale: float = 0.55,
    opt: str = ''
  ):
    image_url = append_query_params(SDSS_RGB_URL, {
      'ra': ra,
      'dec': dec,
      'width': width,
      'height': height,
      'scale': scale,
      'opt': opt
    })
    download_file(image_url, save_path)


  def batch_download_rgb(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Union[str, Path]],
    workers: int = None,
    replace: bool = False,
    **kwargs
  ):
    urls = [
      append_query_params(SDSS_RGB_URL, {'ra': _ra, 'dec': _dec, **kwargs})
      for _ra, _dec in zip(ra, dec)
    ]
    batch_download_file(
      urls=urls,
      save_path=save_path,
      workers=workers,
      replace=replace
    )


  def get_image_filename(
    self,
    dr8objid: np.ndarray,
    dr7objid: np.ndarray,
    extension: str = '.jpg',
    basepath: Union[str, Path] = None
  ) -> Tuple[list, bool]:
    fnames, flag = array_fallback(arrays=(dr7objid, dr8objid), prefix=('dr7', 'dr8'))
    if basepath:
      fnames = [Path(basepath) / f'{f}{extension}' for f in fnames]
    else:
      fnames = [Path(f'{f}{extension}') for f in fnames]
    return fnames, flag


  def crossmatch(
    self,
    ra: Sequence[float],
    dec: Sequence[float],
    fields: Dict[str, Sequence[str]] = {},
    search_type: str = 'photo', # photo or spectro
    search_radius: float = 0.02, # arcmin
    save_path: Union[str, Path] = None,
    chunk_size: int = 500,
    workers: int = 4,
  ) -> pd.DataFrame:
    match_id = secrets.token_hex(4)
    match_folder = Path(f'/tmp/sdss_xmatch_{match_id}')
    match_folder.mkdir(exist_ok=True)

    columns = [f'{k}.{c}' for k, v in fields.items() for c in v['columns']]
    joins = [
      f'{v.get("join_type", "INNER")} JOIN {k} ON {v["join"]}'
      for k, v in fields.items()
    ]

    query = '''
    SELECT {columns}
    FROM #upload AS u
    JOIN #x AS x ON u.up_id = x.up_id
    {joins}
    ORDER BY x.up_id
    '''.strip().format(
      columns=','.join(columns),
      joins='\r\n'.join(joins)
    )

    objects = [f'{_ra},{_dec}' for _ra, _dec in zip(ra, dec)]
    n_chunks = int(np.ceil(len(ra) / chunk_size))
    urls = []

    for i in range(n_chunks):
      if i < n_chunks - 1:
        chunk = objects[chunk_size * i : chunk_size * (i + 1)]
      else:
        chunk = objects[chunk_size * i :]
      params = {
        'searchtool': 'CrossID',
        'searchType': search_type,
        'photoScope': 'nearPrim',
        'spectroScope': 'nearPrim',
        'photoUpType': 'ra-dec',
        'spectroUpType': 'ra-dec',
        'radius': search_radius,
        'firstcol': 0,
        'format': 'csv',
        # 'upload': '',
        'paste': '\r\n'.join(['ra,dec'] + chunk),
        'uquery': query
      }
      urls.append(XID_URL + '?' + urllib.parse.urlencode(params))

    batch_download_file(
      urls=urls,
      save_path=[match_folder / f'chunk_{i}.csv' for i in range(n_chunks)],
      workers=workers,
      replace=False
    )

    combined_csv = pd.concat([
      pd.read_csv(f, comment='#') for f in match_folder.glob('*.csv')
    ])

    if save_path is not None:
      save_path = Path(save_path)
      if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
      combined_csv.to_csv(save_path, index=False)

    return combined_csv



if __name__ == '__main__':
  s = SloanService()
  tb = s.crossmatch(
    ra=[193.78, 199.02, 199.84, 200.44, 208.37, 171.26, 196.62, 199.55, 199.67, 204.95],
    dec=[-1.09, -1.21, -1.13, -1.16, -1.05, -0.81, -0.79, -0.64, -0.74, -0.76],
    fields={
      'SpecObj': {
        'columns': ['z'],
        'join': 'SpecObj.specObjID = x.specObjID',
        'join_type': 'inner'
      },
      'PhotoTag': {
        'columns': ['ra', 'dec', 'modelMag_r'],
        'join': 'PhotoTag.objID = SpecObj.bestObjID',
        'join_type': 'inner'
      }
    },
    search_type='spectro',
    save_path='xmatch.csv',
    workers=3,
    chunk_size=3
  )

  print(tb.columns)

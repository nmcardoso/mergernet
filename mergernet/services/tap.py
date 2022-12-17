import secrets
import tempfile
from pathlib import Path
from typing import List, Union
from urllib.parse import quote, urlencode

import pandas as pd
import requests

from mergernet.services.utils import batch_download_file, download_file


class TapService:
  def __init__(self, url):
    self.url = url
    self.http_client = requests.Session()

  def sync_query(self, query: str, save_path: Union[str, Path]):
    params = {
      'request': 'doQuery',
      'version': 1.0,
      'lang': 'ADQL',
      'phase': 'run',
      'format': 'csv',
      'query': query
    }
    req_url = self.url + '?' + urlencode(params)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(req_url, save_path, replace=True, http_client=self.http_client)

  def batch_sync_query(
    self,
    queries: List[str],
    save_paths: Union[List[Union[str, Path]], str, Path],
    join_outputs: bool = False,
    workers: int = 3
  ):
    params = {
      'request': 'doQuery',
      'version': 1.0,
      'lang': 'ADQL',
      'phase': 'run',
      'format': 'csv'
    }
    urls = [self.url + '?' + urlencode({**params, 'query': q}) for q in queries]

    save_paths_aux = save_paths
    if join_outputs:
      tmp_folder = Path(tempfile.gettempdir()) / f'tap_{secrets.token_hex(3)}'
      tmp_folder.mkdir(parents=True)
      save_paths_aux = [tmp_folder / f'{i}.csv' for i in range(len(queries))]

    batch_download_file(
      urls,
      save_path=save_paths_aux,
      http_client=self.http_client,
      workers=workers
    )

    if join_outputs:
      combined_csv = pd.concat([
        pd.read_csv(f, comment='#') for f in tmp_folder.glob('*.csv')
      ])
      save_paths = Path(save_paths)
      save_paths.parent.mkdir(parents=True, exist_ok=True)
      combined_csv.to_csv(save_paths, index=False)





if __name__ == '__main__':
  tap = TapService('https://datalab.noirlab.edu/tap/sync')
  tap.sync_query('select top 10 psfdepth_g, psfdepth_r from ls_dr9.tractor where ra between 230.2939-0.0013 and 230.2939+0.0013 and dec between 29.7714-0.0013 and 29.7714+0.0013')

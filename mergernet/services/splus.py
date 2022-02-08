from datetime import datetime, timedelta
from enum import Enum
from multiprocessing import Lock
from time import sleep
from types import FunctionType
from typing import List, Union
from pathlib import Path
from urllib.parse import urljoin, quote, urlparse
import concurrent.futures

import requests
import tqdm

from mergernet.core.constants import SPLUS_PASS, SPLUS_USER
from mergernet.services.utils import download_file



BASE_URL = 'https://splus.cloud/api/'
LOGIN_ROUTE = 'auth/login'
LUPTON_ROUTE = 'get_lupton_image/{ra}/{dec}/{size}/{r_band}/{g_band}/{b_band}/{stretch}/{Q}'
TRILOGY_ROUTE = 'get_image/{ra}/{dec}/{size}/{r_band}-{g_band}-{b_band}/{noise}/{saturation}'
FITS_ROUTE = 'get_cut/{ra}/{dec}/{size}/{band}'
PUBLIC_TAP_ROUTE = '/public-TAP/tap/async/?request=doQuery&version=1.0&lang=ADQL&phase=run&query={sql}&format={fmt}'
PRIVATE_TAP_ROUTE = '/public-TAP/tap/async/?request=doQuery&version=1.0&lang=ADQL&phase=run&query={sql}&format={fmt}'



def update_authorization(f):
  def wrapper(*args, **kwargs):
    this: SplusService = args[0]
    updated = this.update_token()
    if updated:
      this.client.headers.update({
        'Authorization': f'Token {this.token["value"]}'
      })
    return f(*args, **kwargs)
  return wrapper



class ImageType(Enum):
  trilogy = 'trilogy'
  lupton = 'lupton'
  fits = 'fits'



class SplusService:
  _lock: Lock = Lock()


  def __init__(self, username: str = SPLUS_USER, password: str = SPLUS_PASS):
    self.credentials = {
      'username': username,
      'password': password
    }
    self.token: dict = None
    self.token_duration = timedelta(hours=1)
    self.client = requests.Session()


  def update_token(self) -> bool:
    with SplusService._lock:
      now = datetime.now()
      if self.token is None or self.token['timestamp'] < (now - self.token_duration):
        resp = self.client.post(
          self._get_url(LOGIN_ROUTE),
          json=self.credentials
        )
        if resp.status_code == 200:
          if 'application/json' in resp.headers['Content-Type']:
            resp_body = resp.json()
            if 'token' in resp_body:
              self.token = {
                'value': resp_body['token'],
                'timestamp': datetime.now()
              }
              return True # updated
    return False # using cache


  def download_lupton_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Union[str, Path],
    replace: bool = False,
    size: int = 128,
    r_band: str = 'I',
    g_band: str = 'R',
    b_band: str = 'G',
    stretch: Union[int, float] = 3,
    Q: Union[int, float] = 8
  ):
    self._download_image(
      LUPTON_ROUTE,
      save_path=save_path,
      replace=replace,
      ra=str(ra),
      dec=str(dec),
      size=str(size),
      r_band=str(r_band),
      g_band=str(g_band),
      b_band=str(b_band),
      stretch=str(stretch),
      Q=str(Q)
    )


  def download_trilogy_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Union[str, Path],
    replace: bool = False,
    size: int = 128,
    r_band: List[str] = ['R', 'I', 'F861', 'Z'],
    g_band: List[str] = ['G', 'F515', 'F660'],
    b_band: List[str] = ['U', 'F378', 'F395', 'F410', 'F430'],
    noise: float = 0.15,
    saturation: float = 0.15
  ):
    self._download_image(
      TRILOGY_ROUTE,
      ra=ra,
      dec=dec,
      save_path=save_path,
      replace=replace,
      size=size,
      r_band=','.join(r_band),
      g_band=','.join(g_band),
      b_band=','.join(b_band),
      noise=noise,
      saturation=saturation
    )


  def download_fits(
    self,
    ra: float,
    dec: float,
    save_path: Union[str, Path],
    replace: bool = False,
    size: int = 128,
    band: str = 'R'
  ):
    self._download_image(
      FITS_ROUTE,
      ra=ra,
      dec=dec,
      save_path=save_path,
      replace=replace,
      size=size,
      band=band
    )


  def batch_image_download(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Union[str, Path]],
    img_type: ImageType = ImageType.lupton,
    workers: int = None,
    **kwargs
  ):
    assert len(ra) == len(dec) == len(save_path)

    if img_type == ImageType.fits:
      download_function = self.download_fits
    elif img_type == ImageType.lupton:
      download_function = self.download_lupton_rgb
    elif img_type == ImageType.trilogy:
      download_function = self.download_trilogy_rgb

    download_args = [
      {'ra': _ra, 'dec': _dec, 'save_path': Path(_save_path), **kwargs}
      for _ra, _dec, _save_path in zip(ra, dec, save_path)
    ]

    self._batch_download(
      download_function=download_function,
      download_args=download_args,
      workers=workers
    )


  def get_image_filename(self):
    pass


  @update_authorization
  def query(
    self,
    sql: str,
    save_path: Union[str, Path],
    replace: bool = False,
    scope: str = 'public',
    fmt: str = 'text/csv'
  ):
    if scope == 'public':
      url = self._get_url(PUBLIC_TAP_ROUTE, {'sql': quote(sql), 'fmt': fmt})
    else:
      url = self._get_url(PRIVATE_TAP_ROUTE, {'sql': quote(sql), 'fmt': fmt})

    resp = self.client.post(
      url,
      headers={
        'Accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    )

    if resp.status_code == 200:
      self._track_tap_job(url=resp.url, save_path=save_path, replace=replace)


  def batch_query(
    self,
    sql: List[str],
    save_path: List[Union[str, Path]],
    replace: bool = False,
    scope: str = 'public',
    fmt: str = 'text/csv',
    workers: int = None
  ):
    assert len(sql) == len(save_path)

    args = [
      {
        'sql': _sql,
        'save_path': _save_path,
        'replace': replace,
        'scope': scope,
        'fmt': fmt
      }
      for _sql, _save_path in zip(sql, save_path)
    ]

    self._batch_download(
      download_function=self.query,
      download_args=args,
      workers=workers
    )


  def _get_url(self, route: str, params: dict = {}) -> str:
    return urljoin(BASE_URL, route.format(**params))


  @update_authorization
  def _track_tap_job(self, url: str, save_path: Union[str, Path], replace: bool):
    resp = self.client.get(url, headers={'Accept': 'application/json'})

    if resp.status_code == 200:
      data = resp.json()
      destruction_time = datetime.fromisoformat(data['destruction'][:-1] + '+00:00')
      now = datetime.now(destruction_time.tzinfo)

      if data['phase'] == 'EXECUTING' and destruction_time > now:
        sleep(5)
        self._track_tap_job(url, save_path, replace)
      elif data['phase'] == 'COMPLETED':
        result_url = urlparse(data['results'][0]['href'])
        result_url = result_url._replace(netloc='splus.cloud').geturl()
        download_file(
          url=result_url,
          save_path=save_path,
          replace=replace,
          http_client=self.client
        )
      elif data['phase'] == 'ERROR':
        message = data['error'].get('message', '')
        print(message)


  @update_authorization
  def _download_image(
    self,
    route: str,
    save_path: Union[str, Path],
    replace: bool = False,
    **kwargs
  ):
    # Stage 1 request
    url = self._get_url(route, kwargs)
    resp = self.client.get(url)

    if resp.status_code == 200:
      if 'application/json' in resp.headers['Content-Type']:
        resp_body = resp.json()
        file_url = self._get_url(resp_body['filename'])
        # Stage 2 request
        download_file(
          file_url,
          save_path=save_path,
          replace=replace,
          http_client=self.client
        )


  def _batch_download(
    self,
    download_function: FunctionType,
    download_args: dict,
    workers: int = None
  ):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
      futures = []
      download_function = download_function or download_file

      for i in range(len(download_args)):
        futures.append(executor.submit(
          download_function,
          **download_args[i]
        ))

      for future in tqdm.tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        unit=' files'
      ):
        future.result()



if __name__ == '__main__':
  s = SplusService()
  ra = [11.5933851, 11.8345742, 11.9053378, 12.1397573, 12.3036425]
  dec = [-1.0180862, -0.8710110, -0.8707459, -0.7373196, -0.4088959]
  path1 = ['1.png', '2.png', '3.png', '4.png', '5.png']
  path2 = ['01.png', '02.png', '03.png', '04.png', '05.png']
  path3 = ['1.fits', '2.fits', '3.fits', '4.fits', '5.fits']
  # s.batch_image_download(ra, dec, path1, img_type=ImageType.lupton, workers=3, replace=True)
  # s.batch_image_download(ra, dec, path2, img_type=ImageType.trilogy, workers=3, replace=True)
  # s.batch_image_download(ra, dec, path3, img_type=ImageType.fits, workers=3, replace=True)


  sql = ['SELECT TOP 100 ID, RA, DEC FROM dr3.all_dr3 where id like \'%HYDRA%\'']
  # path4 = [f'table{i}.csv' for i in range(10)]
  path4 = ['table0.csv']
  s.batch_query(sql, save_path=path4, replace=True, workers=2)

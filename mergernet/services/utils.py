from pathlib import Path
from types import FunctionType
from typing import List, Union
import concurrent.futures

import tqdm
import requests



def download_file(
  url: str,
  save_path: Union[str, Path],
  replace: bool = False,
  http_client: requests.Session = None
):
  save_path = Path(save_path)

  if not replace and save_path.exists():
    return

  http_client = http_client or requests

  if not save_path.parent.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)

  r = http_client.get(url, allow_redirects=True)

  with open(str(save_path.resolve()), 'wb') as f:
    f.write(r.content)



def batch_download_file(
  urls: List[str],
  save_path: List[Union[str, Path]],
  workers: int = 2,
  replace: bool = False
):
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = []

    for i in range(len(urls)):
      futures.append(executor.submit(
        download_file,
        url=urls[i],
        save_path=Path(save_path[i]),
        replace=replace
      ))

    for _ in tqdm.tqdm(
      concurrent.futures.as_completed(futures),
      total=len(futures),
      unit=' files'
    ):
      pass



def append_query_params(url: str, query_params: dict) -> str:
  index = url.find('?')
  new_url = ''

  query_string = '&'.join([f'{k}={v}' for (k, v) in query_params.items()])

  if index > -1:
    base = url[:index]
    rest = url[index:] + '&' + query_string
    new_url = base + rest
  else:
    new_url = url + '?' + query_string

  return new_url

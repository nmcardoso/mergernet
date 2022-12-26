import bz2
import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

import requests
import tqdm


def parallel_function_executor(
  func: Callable,
  params: Sequence[Dict[str, Any]] = [],
  workers: int = 2,
  unit: str = ' it'
):
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = []

    for i in range(len(params)):
      futures.append(executor.submit(
        func,
        **params[i]
      ))

    for future in tqdm.tqdm(
      concurrent.futures.as_completed(futures),
      total=len(futures),
      unit=unit
    ):
      try:
        future.result()
      except Exception as e:
        pass



def download_file(
  url: str,
  save_path: Union[str, Path],
  replace: bool = False,
  http_client: requests.Session = None,
  extract: bool = False
):
  save_path = Path(save_path)

  if not replace and save_path.exists():
    return

  http_client = http_client or requests

  if not save_path.parent.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)

  r = http_client.get(url, allow_redirects=True)

  if r.status_code == 200:
    if extract:
      file_bytes = bz2.decompress(r.content)
    else:
      file_bytes = r.content

    with open(str(save_path.resolve()), 'wb') as f:
      f.write(file_bytes)



def batch_download_file(
  urls: List[str],
  save_path: List[Union[str, Path]],
  replace: bool = False,
  http_client: requests.Session = None,
  workers: int = 2,
):
  params = [
    {
      'url': _url,
      'save_path': Path(_save_path),
      'replace': replace,
      'http_client': http_client
    }
    for _url, _save_path in zip(urls, save_path)
  ]
  parallel_function_executor(download_file, params, workers=workers, unit=' files')



# def batch_download_file(
#   urls: List[str],
#   save_path: List[Union[str, Path]],
#   workers: int = 2,
#   replace: bool = False
# ):
#   with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
#     futures = []

#     for i in range(len(urls)):
#       futures.append(executor.submit(
#         download_file,
#         url=urls[i],
#         save_path=Path(save_path[i]),
#         replace=replace
#       ))

#     for _ in tqdm.tqdm(
#       concurrent.futures.as_completed(futures),
#       total=len(futures),
#       unit=' files'
#     ):
#       pass



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

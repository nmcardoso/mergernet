from pathlib import Path
from typing import List
import concurrent.futures

import tqdm
import requests



def download_file(url: str, save_path: Path, replace: bool = False):
  if not replace and save_path.exists():
    return

  if not save_path.parent.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)

  r = requests.get(url, allow_redirects=True)

  with open(str(save_path.resolve()), 'wb') as f:
    f.write(r.content)



def batch_download_file(
  urls: List[str],
  save_path: List[Path],
  workers: int = 2,
  replace: bool = False
):
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = []

    for i in range(len(urls)):
      futures.append(executor.submit(
        download_file,
        url=urls[i],
        save_path=save_path[i],
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

  if index > -1:
    base = url[:index]
    rest = url[index:] + '&' + '&'.join([f'{k}={v}' for (k, v) in query_params.items()])
    new_url = base + rest
  else:
    new_url = url + '?' + '&'.join([f'{k}={v}' for (k, v) in query_params.items()])

  return new_url

from pathlib import Path

import requests



  if not save_path.parent.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)
  r = requests.get(remote_url, allow_redirects=True)
  with open(str(save_path.resolve()), 'wb') as f:
    f.write(r.content)



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

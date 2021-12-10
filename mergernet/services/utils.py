from pathlib import Path

import requests



def download_file(remote_url: str, save_path: Path):
  if not Path.exists(save_path.parent):
    save_path.parent.mkdir(parents=True, exist_ok=True)
  r = requests.get(remote_url, allow_redirects=True)
  with open(str(save_path.resolve()), 'wb') as f:
    f.write(r.content)



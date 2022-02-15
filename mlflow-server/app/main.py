from typing import Union
from pathlib import Path
from shutil import make_archive

from aiohttp import web
from aiohttp import streamer



@streamer
async def file_sender(writer, file_path: Union[str, Path] = None):
  """
  This function will read large file chunk by chunk and send it through HTTP
  without reading them into memory
  """
  with open(file_path, 'rb') as f:
    chunk = f.read(2 ** 16)
    while chunk:
      await writer.write(chunk)
      chunk = f.read(2 ** 16)



async def get_database_backup(request):
  headers = {
    "Content-disposition": "attachment; filename=mlflow.sqlite"
  }

  file_path = Path('/data/app/mlflow.sqlite')

  if not file_path.exists():
    return web.Response(
      body='File <mlflow.sqlite> does not exist',
      status=404
    )

  return web.Response(
    body=file_sender(file_path=file_path),
    headers=headers
  )



async def get_artifacts_backup(request):
  path = Path('/app/data/artifacts')
  if not path.exists():
    path.mkdir(parents=True)
    with open(path / 'test.txt', 'w') as f:
      f.write('test')

  make_archive(
    base_name='/app/data/artifacts.tar.gz',
    format='gztar',
    root_dir='/app/data/artifacts'
  )

  headers = {
    "Content-disposition": "attachment; filename=artifacts.tar.gz"
  }

  file_path = Path('/app/data/artifacts.tar.gz')

  if not file_path.exists():
    return web.Response(
      body='File <artifacts.tar.gz> does not exist',
      status=404
    )

  return web.Response(
    body=file_sender(file_path=file_path),
    headers=headers
  )




if __name__ == '__main__':
  app = web.Application()
  app.add_routes([
    web.get('/backup/db', get_database_backup),
    web.get('/backup/artifacts', get_artifacts_backup)
  ])
  web.run_app(app, host='0.0.0.0', port=8081)

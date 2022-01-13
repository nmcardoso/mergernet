from mergernet.jobs import *
from importlib import import_module



DB = []

def dec(f):
  print('decorator')
  print(f)
  i = f()
  i.run()
  DB.append(f)
  return f



m = import_module(f'..jobs.j001_download_imgs', package='mergernet.jobs')


if __name__ == '__main__':
  print(m)

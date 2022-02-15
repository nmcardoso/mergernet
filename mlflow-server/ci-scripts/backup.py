from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

import os
from pathlib import Path


gauth = GoogleAuth(settings_file='google_settings.yaml')

drive = GoogleDrive(gauth)

path = Path('backup.py')

q = "'root' in parents"
file_list = drive.ListFile({'q': q}).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))

# f = drive.CreateFile({'title': path.name})
# f.SetContentFile(str(path.resolve()))
# f.Upload()
# f = None

# for x in path.iterdir():
#   print(x)
#   f = drive.CreateFile({'title': x.stem})
#   f.SetContentFile(str(x.resolve()))
#   f.Upload()
#   f = None # fix pydrive bug that causes memory leak

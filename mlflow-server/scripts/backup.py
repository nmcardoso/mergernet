import os
import json
from datetime import timezone, datetime, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaFileUpload
import requests


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']


def download_file(url, path):
  r = requests.get(url, allow_redirects=True)
  with open(path, 'wb') as f:
    f.write(r.content)


def get_credentials():
  creds = None
  if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      creds = ServiceAccountCredentials.from_json_keyfile_name(
        'credentials.json',
        scopes=SCOPES
      )
    with open('token.json', 'w') as token:
      token.write(creds.to_json())
  return creds


def upload_file(path, mimetype, service):
  fname = Path(path).name
  file_metadata = {'name': fname}
  media = MediaFileUpload(path, mimetype=mimetype)

  file = service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id'
  ).execute()

  print('File ID: %s' % file.get('id'))

  permission_info = service.permissions().create(
    fileId=file.get('id'),
    body={'role': 'reader', 'type': 'anyone'}
  ).execute()

  file_info = service.files().get(
    fileId=file.get('id'),
    fields=','.join([
      'id', 'name', 'mimeType', 'webContentLink', 'webViewLink', 'createdTime',
      'modifiedTime', 'shared', 'permissionIds', 'permissions', 'size',
      'fullFileExtension', 'originalFilename'
    ])
  ).execute()

  with open(f'../backups/{fname}.json', 'w') as fp:
    json.dump(file_info, fp, indent=True)


def main():
  # timestamp
  tz = timezone(timedelta(hours=-3))
  d = datetime.now(tz)
  ts = '{}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
    d.year, d.month, d.day, d.hour, d.minute, d.second
  )

  # download database
  db_fname = f'mlflow_{ts}.sqlite'
  print(f'Downloading {db_fname}')
  download_file('http://backup-nmcardoso.cloud.okteto.net/db', f'/tmp/{db_fname}')

  # download artifacts
  artifacts_fname = f'artifacts_{ts}.tar.gz'
  print(f'Downloading {artifacts_fname}')
  download_file('http://backup-nmcardoso.cloud.okteto.net/artifacts', f'/tmp/{artifacts_fname}')

  # setup credentials
  print('Generating Google credentials')
  creds = get_credentials()

  try:
    service = build('drive', 'v3', credentials=creds)

    # upload database
    print(f'Uploading {db_fname}')
    upload_file(f'/tmp/{db_fname}', 'application/vnd.sqlite3', service)

    # upload artifacts
    print(f'Uploading {artifacts_fname}')
    upload_file(f'/tmp/{artifacts_fname}', 'application/gzip', service)
  except HttpError as error:
    print(f'An error occurred: {error}')


if __name__ == '__main__':
  main()

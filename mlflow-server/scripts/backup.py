import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaFileUpload


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']


def main():
  creds = None

  if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      creds = ServiceAccountCredentials.from_json_keyfile_name(
        'credentials.json', scopes=SCOPES
      )

    with open('token.json', 'w') as token:
      token.write(creds.to_json())

  try:
    service = build('drive', 'v3', credentials=creds)

    # upload file t

    # upload
    # file_metadata = {'name': 'img.gif'}
    # media = MediaFileUpload(f'img.gif', mimetype='image/gif')
    # file = service.files().create(
    #   body=file_metadata,
    #   media_body=media,
    #   fields='id'
    # ).execute()
    # print('File ID: %s' % file.get('id'))
    # r1 = service.permissions().create(fileId=file.get('id'), body={'role': 'reader', 'type': 'anyone'}).execute()
    # print(r1)

    # r2 = service.files().get(fileId=file.get('id'), fields='*').execute()
    # print(r2)

    # list
    results = service.files().list(
      q="'root' in parents", pageSize=60, spaces='drive').execute()
    items = results.get('files', [])


    if not items:
      print('No files found.')
      return
    print('Files:')
    for item in items:
      print(u'{0} ({1})'.format(item['name'], item['id']))
      r = service.files().delete(fileId=item['id']).execute()
      print(r)
  except HttpError as error:
    print(f'An error occurred: {error}')


if __name__ == '__main__':
  main()

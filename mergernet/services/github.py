import base64

import requests


BASE_URL = 'https://api.github.com'
HEADERS = {
  'Accept': 'application/vnd.github.v3+json'
}


class GithubService:
  def __init__(self, user: str, token: str, repo: str):
    self.user = user
    self.token = token
    self.repo = repo


  def _get_url(self, route: str) -> str:
    return f'{BASE_URL}/{route}'

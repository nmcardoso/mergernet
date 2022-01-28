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


  def _encode_content(self, content: str) -> str:
    content_bytes = content.encode(encoding='utf-8')
    base64_bytes = base64.b64encode(content_bytes)
    base64_str = base64_bytes.decode('utf-8')
    return base64_str


  def commit(self, path: str, data: str, branch: str):
    url = self._get_url(f'repos/{self.user}/{self.repo}/contents/{path}')

    commit_data = {
      'message': 'test commit',
      'content': self._encode_content(data),
      'branch': branch
    }

    response = requests.get(
      url=url,
      headers=HEADERS,
      auth=(self.user, self.token)
    )

    print(response.json())

    response_data = response.json()
    if 'sha' in response_data:
      commit_data['sha'] = response_data['sha']

    response = requests.put(
      url=url,
      headers=HEADERS,
      json=commit_data,
      auth=(self.user, self.token)
    )

    print(response.json())


  def list_dir(self, path: int) -> dict:
    url = self._get_url(f'repos/{self.user}/{self.repo}/contents/{path}')

    response = requests.get(
      url=url,
      headers=HEADERS,
      auth=(self.user, self.token)
    )

    return response.json()


  def get_lastest_job_run(self, jobid: int) -> Union[int, None]:
    content = self.list_dir('jobs_artifacts')
    folders = [c['name'] for c in content if c['type'] == 'dir']

    runs = []
    exp = r'job_{0:0=3d}_run_(\d+)'.format(jobid)
    for folder in folders:
      match = re.search(exp, folder)
      if match:
        runs.append(int(match[1]))

    if len(runs) > 0:
      return max(runs)
    else:
      return None






if __name__ == '__main__':
  gh_service = GithubService('nmcardoso', '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', 'arial')
  gh_service.commit('filename.text', 'Caracas2', 'master')

import os
from pathlib import Path

# Check env
ENV = os.environ.get('PYTHON_ENV', 'dev')



# Commom paths
SOURCE_ROOT = (Path(__file__).parent.parent).resolve()
PROJECT_ROOT = SOURCE_ROOT.parent
DOCS_ROOT = PROJECT_ROOT / 'docs'
DATA_ROOT = PROJECT_ROOT / 'data'



# Github authentication
GITHUB_USER = os.environ.get('GITHUB_USER', None)
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', None)
GITHUB_REPO = os.environ.get('GITHUB_REPO', None)
GITHUB_PATH = os.environ.get('GITHUB_PATH', 'data/jobs_artifacts')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')



# Google Drive
if ENV == 'dev':
  GDRIVE_PATH = 'gdrive'
else:
  GDRIVE_PATH = os.environ.get('GDRIVE_PATH', None)



# Random number generation seed
RANDOM_SEED = 42

import os
from pathlib import Path

# Check env
ENV = os.environ.get('PYTHON_ENV', 'dev')



# S-PLUS Bands
SPLUS_BANDS = [
  'R', 'I', 'F861', 'Z', 'G', 'F515',
  'F660', 'U', 'F378', 'F395', 'F410', 'F430'
]



# Commom paths
SOURCE_ROOT = (Path(__file__).parent.parent).resolve()
PROJECT_ROOT = SOURCE_ROOT.parent
DOCS_ROOT = PROJECT_ROOT / 'docs'
DATA_ROOT = PROJECT_ROOT / 'data'
JOBS_PATH = SOURCE_ROOT / 'jobs'
SAVED_MODELS_PATH = Path('saved_models')



# Github authentication
GITHUB_USER = os.environ.get('GITHUB_USER', None)
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', None)
GITHUB_REPO = os.environ.get('GITHUB_REPO', None)
GITHUB_PATH = os.environ.get('GITHUB_PATH', 'jobs_artifacts')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')



# S-PLUS authentication
SPLUS_USER = os.environ.get('SPLUS_USER', None)
SPLUS_PASS = os.environ.get('SPLUS_PASS', None)



# Google Drive
GDRIVE_PATH = os.environ.get('GDRIVE_PATH', None)
if ENV == 'dev' and not GDRIVE_PATH:
  GDRIVE_PATH = DATA_ROOT / 'gdrive'



# Random number generation seed
RANDOM_SEED = 42



# MLflow default database name
MLFLOW_DEFAULT_DB = 'mlflow.sqlite'
MLFLOW_DEFAULT_URL = 'https://mlflow-nmcardoso.cloud.okteto.net'

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



# Github authentication and paths
GH_USER = os.environ.get('GITHUB_USER', None)
GH_TOKEN = os.environ.get('GITHUB_TOKEN', None)
GH_REPO = os.environ.get('GITHUB_REPO', 'mergernet-artifacts')
GH_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')
GH_BASE_PATH = os.environ.get('GITHUB_PATH', '')
GH_ARTIFACT_PATH = GH_BASE_PATH + '/jobs/{job_id}/{run_id}/{model_name}/{artifact_name}'
GH_OPTUNA_PATH = GH_BASE_PATH + '/optuna/{artifact_name}'



# Google Drive paths
GDRIVE_MOUNT_PATH = os.environ.get('GDRIVE_PATH', None)
if ENV == 'dev' and not GDRIVE_MOUNT_PATH:
  GDRIVE_MOUNT_PATH = DATA_ROOT / 'gdrive'
GDRIVE_ARTIFACT_PATH = str(GDRIVE_MOUNT_PATH) + '/jobs/{job_id}/{run_id}/{artifact_name}'



# S-PLUS authentication
SPLUS_USER = os.environ.get('SPLUS_USER', None)
SPLUS_PASS = os.environ.get('SPLUS_PASS', None)



# Random number generation seed
RANDOM_SEED = 42

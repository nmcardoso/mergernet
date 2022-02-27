from pathlib import Path
import sys
import os
import shutil
project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

from mergernet.services.splus import SplusService


sql_file = '../data/sql/splus_stripe.sql'
with open(sql_file) as fp:
  sql = fp.read()

splus = SplusService()
splus.batch_query([sql], save_path=['../data/splus_south_gal80.csv'], scope='private')

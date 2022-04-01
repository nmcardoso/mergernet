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
splus.batch_query([sql], save_path=['../data/blind_splus_gal80_r13.5-17.csv'], scope='private')

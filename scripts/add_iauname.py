import argparse
import os
import sys
from pathlib import Path
from typing import Union

import pandas as pd

project_root = str(Path(os.path.abspath('')).parent.resolve())
sys.path.append(project_root)

from mergernet.core.utils import iauname




def main(
  path: Union[str, Path],
  ra_col: str,
  dec_col: str,
  save_path: Union[str, Path]
):
  df = pd.read_csv(path)
  ra = df[ra_col].to_numpy()
  dec = df[dec_col].to_numpy()
  df['iauname'] = iauname(ra, dec)
  # df['iauname'] = [iauname(_ra, _dec) for _ra, _dec in zip(ra, dec)]
  df.to_csv(save_path, index=False)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', action='store', help='The input table path', default=None)
  parser.add_argument('--ra', action='store', help='The column name of RA in the table', default=None)
  parser.add_argument('--dec', action='store', help='The column name of DEC in the table', default=None)
  parser.add_argument('-o', action='store', help='The output table path', default=None)
  args = parser.parse_args()
  main(args.t, args.ra, args.dec, args.o)

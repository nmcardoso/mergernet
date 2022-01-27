from pathlib import Path
from typing import Union

from mergernet.core.constants import DATA_ROOT

from astropy.io import fits
from astropy.table import Table
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf



def load_image(path: Path) -> np.ndarray:
  """Load image from local storage to numpy array.
  Supports several types of files, incluing ``.jpg``, ``.png``, ``.npy``,
  ``.npz``, ``.fits``

  Parameters
  ----------
  path: pathlib.Path
    Path to the desired file

  Returns
  -------
  numpy.ndarray
    Image converted (if needed) to a numpy array.
  """
  if path.suffix in ['.jpg', '.png']:
    img = Image.open(path)
    return tf.keras.preprocessing.image.img_to_array(img)
  elif path.suffix in ['.npy', '.npz']:
    return np.load(path)
  elif path.suffix == '.fits':
    pass



def load_table(path: Union[Path, str], default: bool = True) -> pd.DataFrame:
  if default:
    path = DATA_ROOT / 'tables' / path

  if path.suffix in {'.fit', '.fits'}:
    with fits.open(path) as hdul:
      table_data = hdul[1].data
      table = Table(data=table_data)
    return table.to_pandas()
  elif path.suffix == '.csv':
    return pd.read_csv(path)



def save_table(data: pd.DataFrame, path: Union[Path, str], default: bool = True):
  if default:
    path = DATA_ROOT / 'tables' / path

  if path.suffix in {'.fit', '.fits'}:
    pass
  elif path.suffix == '.csv':
    data.to_csv(path, index=False)



def array_fallback(arrays, prefix=None):
  gen = []
  flag = False # return True if missing row

  for i in range(arrays[0].shape[0]):
    flag_aux = False

    for j in range(len(arrays)):
      row = arrays[j][i]
      if row:
        flag_aux = True
        if prefix:
          gen.append(f'{prefix[j]}_{row}')
        else:
          gen.append(row)
        break

    if not flag_aux:
      flag = True

  return gen, flag




class Timming:
  def __init__(self):
    self.start_time = None
    self.end_time = None


  def __repr__(self) -> str:
    return self.duration()


  def start(self):
    self.start_time = datetime.now()


  def end(self):
    self.end_time = datetime.now()


  def duration(self) -> str:
    if not self.end_time:
      duration = self.end_time - self.start_time
    else:
      end_time = datetime.now()
      duration = end_time - self.start_time

    return self._format_time(duration)


  def _format_time(self, dt: datetime) -> str:
    r = ''

    if dt.day:
      r += f'{dt.day}d'

    if dt.hour:
      r += f'{dt.hour}h'

    if dt.minute:
      r += f'{dt.minute}m'

    if dt.second:
      r += f'{dt.second}s'

    if r == '':
      r = '0s'

    return r



class SingletonMeta(type):
  """The Singleton class can be implemented in different ways in Python. Some
  possible methods include: base class, decorator, metaclass. We will use the
  metaclass because it is best suited for this purpose.
  """

  _instances = {}

  def __call__(cls, *args, **kwargs):
    """Possible changes to the value of the `__init__` argument do not affect
    the returned instance.
    """
    if cls not in cls._instances:
      instance = super().__call__(*args, **kwargs)
      cls._instances[cls] = instance
    return cls._instances[cls]

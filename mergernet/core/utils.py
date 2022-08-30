import collections.abc
import json
from pathlib import Path
from typing import Any, Sequence, Union
from datetime import datetime, timedelta
from threading import Lock

from mergernet.core.constants import DATA_ROOT, ENV

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import ICRS, SkyCoord
from astropy import units as u
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf



def load_image(path: Union[str, Path]) -> np.ndarray:
  """
  Load image from local storage to numpy array.
  Supports several types of files, incluing ``.jpg``, ``.png``, ``.npy``,
  ``.npz``, ``.fits``

  Parameters
  ----------
  path: str or pathlib.Path
    Path to the desired file

  Returns
  -------
  numpy.ndarray
    Image converted (if needed) to a numpy array.
  """
  path = Path(path)
  if path.suffix in ['.jpg', '.png']:
    img = Image.open(path)
    return np.asarray(img)
  elif path.suffix in ['.npy', '.npz']:
    return np.load(path)
  elif path.suffix in ['.fits', '.fit', '.fz']:
    return fits.getdata(path)



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



def deep_update(original, new, *args):
  updates = (new,) + args
  for u in updates:
    for k, v in u.items():
      if isinstance(v, collections.abc.Mapping):
        original[k] = deep_update(original.get(k, {}), v)
      else:
        original[k] = v
    return original



def unique_path(path: Union[str, Path]):
  path = Path(path)
  new_path = path
  i = 2
  while(not new_path.exists()):
    parent = new_path.parent
    ext = ''.join(new_path.suffixes)
    name = new_path.name
    new_name = name + '_' + str(i)
    new_path = parent / (new_name + ext)
    i += 1
  return new_path



def find_by_value(d: dict, search: Any):
  """
  Search a dictionary by values and returns the corresponding key.

  Parameters
  ----------
  d: dict
    Dictionary to search.

  search: Any
    Value of the dictionary to search.

  Returns
  -------
  Any
    The key of corresponding value.
  """
  return list(d.keys())[list(d.values()).index(search)]



def iauname(
  ra: Union[float, np.ndarray],
  dec: Union[float, np.ndarray]
) -> Union[str, List[str]]:
  """
  Receives the angular position(s) of the object(s) and returns IAU2000 name(s)

  Parameters
  ----------
  ra: float or array of float
    The right ascension of the object(s).
  dec: float or array of float
    The declination of the object(s).

  Example
  --------
  >>> iauname(187.70592, 12.39112)
  'J123049.42+122328.03'

  Returns
  -------
  str or list of str
    The formated IAU name of the object(s)
  """
  coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
  ra_str = coord.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
  dec_str = coord.dec.to_string(sep='', precision=1, alwayssign=True, pad=True)
  if isinstance(ra_str, np.ndarray):
    r = [f'J{_ra_str}{_dec_str}' for _ra_str, _dec_str in zip(ra_str, dec_str)]
  else:
    r = f'J{ra_str}{dec_str}'
  return r



def skip_dev(func):
  """
  Decorator used to skip high complexity functions/methods in development
  environment with pythonic syntax suggar
  """
  if ENV == 'dev':
    return lambda *args, **kwargs: 0 # null func
  else:
    return func



class Timming:
  def __init__(self, start: bool = True):
    self.start_time = None
    self.end_time = None
    if start:
      self.start()


  def __repr__(self) -> str:
    return self.duration()


  def start(self):
    self.start_time = datetime.now()


  def end(self) -> str:
    self.end_time = datetime.now()
    return self.duration()


  def duration(self) -> str:
    if not self.end_time:
      duration = self.end_time - self.start_time
    else:
      end_time = datetime.now()
      duration = end_time - self.start_time

    return self._format_time(duration)


  def _format_time(self, dt: timedelta) -> str:
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))



class SingletonMeta(type):
  """
  Thread-safe implementation of Singleton.
  """

  _instances = {}

  _lock: Lock = Lock()
  """
  Lock object that will be used to synchronize threads during
  first access to the Singleton.
  """

  def __call__(cls, *args, **kwargs):
    """
    Possible changes to the value of the `__init__` argument do not affect
    the returned instance.
    """
    # When the program has just been launched. Since there's no
    # Singleton instance yet, multiple threads can simultaneously pass the
    # previous conditional and reach this point almost at the same time. The
    # first of them will acquire lock and will proceed further, while the
    # rest will wait here.
    with cls._lock:
      # The first thread to acquire the lock, reaches this conditional,
      # goes inside and creates the Singleton instance. Once it leaves the
      # lock block, a thread that might have been waiting for the lock
      # release may then enter this section. But since the Singleton field
      # is already initialized, the thread won't create a new object.
      if cls not in cls._instances:
        instance = super().__call__(*args, **kwargs)
        cls._instances[cls] = instance
    return cls._instances[cls]



def serialize(obj: Any) -> str:
  """
  Serializes an object performing type cast depending of the object type

  Parameters
  ----------
  obj: Any
    The object to be serialized

  Returns
  -------
  str
    The serialized object
  """
  def to_primitive(v):
    if isinstance(v, np.ndarray):
      return v.tolist()
    elif isinstance(v, Path):
      return str(v)
    else:
      return v

  prepared_obj = {
    k: to_primitive(v) for k, v in obj.items()
  }

  return json.dumps(prepared_obj)

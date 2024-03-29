import collections.abc
import json
import logging
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, List, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from PIL import Image, ImageOps
from tqdm import tqdm

from mergernet.core.constants import DATA_ROOT, ENV, RANDOM_SEED

L = logging.getLogger(__name__)


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
  if path.suffix in ('.jpg', '.png'):
    img = Image.open(path)
    return np.asarray(img)
  elif path.suffix in ('.npy', '.npz'):
    return np.load(path)
  elif path.suffix in ('.fits', '.fit', '.fz'):
    im = fits.getdata(path)
    if np.argmin(im.shape) == 0:
      # convert from (c, h, w) to (h, w, c)
      im = np.moveaxis(im, 0 , -1)
    return im



def save_image(data: np.ndarray, path: Union[str, Path]):
  path = Path(path)

  if not path.parent.exists():
    path.parent.mkdir(parents=True, exist_ok=True)

  if path.suffix in ('.jpg', '.png'):
    ImageOps.flip(Image.fromarray(data, 'RGB')).save(path)
  elif path.suffix in ('.fits', '.fit'):
    fits.ImageHDU(np.moveaxis(data, -1, 0)).writeto(path, overwrite=True)
  elif path.suffix == '.npy':
    np.save(path, data)
  elif path.suffix == '.npz':
    np.savez(path, *data)



def load_table(path: Union[Path, str], default: bool = False) -> pd.DataFrame:
  if default:
    path = DATA_ROOT / 'tables' / path

  path = Path(path)

  if path.suffix in ('.fit', '.fits', '.fz'):
    with fits.open(path) as hdul:
      table_data = hdul[1].data
      table = Table(data=table_data)
    return table.to_pandas()
  elif path.suffix == '.csv':
    return pd.read_csv(path)
  elif path.suffix == '.parquet':
    return pd.read_parquet(path)



def save_table(data: pd.DataFrame, path: Union[Path, str], default: bool = True):
  if default:
    path = DATA_ROOT / 'tables' / path

  path = Path(path)

  if path.suffix in ('.fit', '.fits'):
    Table.from_pandas(data).write(path, overwrite=True)
  elif path.suffix == '.csv':
    data.to_csv(path, index=False)
  elif path.suffix == '.parquet':
    data.to_parquet(path, index=False)



def compress_fits(
  file: Union[str, Path, BinaryIO],
  compress_type: str = 'HCOMPRESS_1',
  hcomp_scale: int = 3,
  quantize_level: int = 10,
  quantize_method: int = -1,
  ext: int = 0,
  save_path: Union[str, Path] = None,
  replace: bool = True,
):
  hdul = fits.open(file)

  if ext >= len(hdul):
    raise ValueError(f'Trying to access ext {ext}, max ext is {len(hdul)-1}')

  if save_path.exists() and not replace:
    return None

  comp = None

  try:
    comp = fits.CompImageHDU(
      data=hdul[ext].data,
      header=hdul[ext].header,
      compression_type=compress_type,
      hcomp_scale=hcomp_scale,
      quantize_level=quantize_level,
      quantize_method=quantize_method,
      dither_seed=RANDOM_SEED,
    )
    if save_path:
      comp.writeto(
        save_path,
        overwrite=replace,
      )
  except OSError as e:
    pass

  return comp



def extract_iauname_from_path(path: Path):
  iauname = path.stem
  if iauname.endswith('.fits'):
    iauname = iauname[:-5]
  return iauname



def execute_posix_command(command: str):
  L.info(f'executing posix command: {command}')

  process = subprocess.Popen(
    command,
    shell=True,
    stdout=subprocess.PIPE,
    executable='/bin/bash'
  )

  while True:
    output = process.stdout.readline()
    if (output == b'' or output == '') and process.poll() is not None:
      break
    else:
      print(output)



def install_linux_package(package: str):
  if shutil.which(package) is None:
    execute_posix_command(f'sudo apt install {package}')



def compress_files(
  input_path: Union[str, Path, List[str], List[Path]],
  output_path: Union[str, Path],
  level: int = 2,
):
  if isinstance(input_path, list):
    base_path = Path(input_path[0]).parent
    input_path = [Path(p) for p in input_path]
    input_path_str = [str(p.resolve()) for p in input_path]
    folders = [p.name for p in input_path]
  else:
    base_path = Path(input_path).parent
    input_path = Path(input_path)
    input_path_str = [str(input_path.resolve())]
    folders = [input_path.name]

  output_path = Path(output_path)

  install_linux_package('pv')

  command = (
    f"tar -C {str(base_path.resolve())} -cf - {' '.join(folders)} -P 2>/dev/null | "
    f"pv -s $(du -sbc {' '.join(input_path_str)} 2>/dev/null | awk 'END{{print $1}}') | "
    f"xz -{level} > {str(output_path.resolve())}"
  )

  execute_posix_command(command)



def extract_files(archive: Union[str, Path], dest: Union[str, Path]):
  archive = Path(archive)
  dest = Path(dest)

  install_linux_package('pv')

  command = f'pv {str(archive.resolve())} | tar -xJ -C {str(dest.resolve())}'

  dest.mkdir(parents=True, exist_ok=True)

  execute_posix_command(command)



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



def iauname_path(
  iaunames: Union[str, List[str]] = None,
  ra: Union[float, List[float]] = None,
  dec: Union[float, List[float]] = None,
  prefix: Union[str, Path] = '',
  suffix: str = '',
  flat: bool = False,
  return_str: bool = False,
) -> Union[Path, List[Path]]:
  """
  Calculate the nested path for a given iauname

  Parameters
  ----------
  iaunames: str, List[str], optional
    Object iauname. The iauname or RA and DEC must be passed, if ``iaunames`` is
    ``None``, this function computes the iauname using the ``ra`` and ``dec``
    parameters
  ra: float, List[float], optional
    Object RA, used only if ``iaunames`` is ``None``
  dec: float, List[float], optional
    Object DEC, used only if ``iaunames`` is ``None``
  prefix: str, Path, optional
    Path that will be prepended at the begin of all paths
  suffix: str, optional
    Suffix that will be appended at the end of all paths
  flat: bool, optional
    Create the flat path with all files inside a same parent folder. This is
    not recomended for big datasets
  return_str: bool, optional
    Cast all paths to string before returning

  Example
  -------
  iaunames_path('J123049.42+122328.03', '.png')
  Path('J123/J123049.42+122328.03.png')

  Returns
  -------
  Path, List[Path]
    The iauname path
  """
  if iaunames is None:
    iaunames = iauname(ra, dec)

  if flat:
    mapping = lambda x:  Path(prefix) / (x + suffix)
  else:
    mapping = lambda x: Path(prefix) / x[:4] / (x + suffix)

  prep_output = lambda x: str(x) if return_str else x

  if isinstance(iaunames, str):
    return prep_output(mapping(iaunames))
  else:
    return [prep_output(mapping(x)) for x in iaunames]



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



def heading(msg: str, sep: str = '-'):
  """
  Prints a message with a rule bellow with same width of the message

  Parameters
  ----------
  msg: str
    The message string
  sep: str
    The rule character
  """
  print(msg)
  print(sep*len(msg))



def not_in(ref: list, comp: list):
  """
  Computes elements in ``ref`` that are not in ``comp``

  Parameters
  ----------
  ref: list
    The reference list
  comp: list
    The comparison list

  Returns
  -------
  list
    A list containing all elements in ``ref`` that are not in ``comp``
  """
  return list(set(ref) - set(comp))



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



class CachedDataFrame:
  """
  Holds a cached version of a pandas dataframe in memory for later access
  """
  _cache = {}

  @classmethod
  def load(cls, path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a dataframe in memory

    Parameters
    ----------
    path: str or Path
      The path of the dataframe

    Returns
    -------
    pd.DataFrame
      The cached/loaded dataframe
    """
    path_str = str(path)
    if path_str not in cls._cache:
      cls._cache[path_str] = pd.read_csv(path, comment='#')
    return cls._cache[path_str]



class classproperty(property):
  def __get__(self, owner_self, owner_cls):
    return self.fget(owner_cls)


# class classproperty(property):
#   def __get__(self, obj, objtype=None):
#     return super(classproperty, self).__get__(objtype)
#   def __set__(self, obj, value):
#     super(classproperty, self).__set__(type(obj), value)
#   def __delete__(self, obj):
#     super(classproperty, self).__delete__(type(obj))

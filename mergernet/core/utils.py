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


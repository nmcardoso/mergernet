"""
Dataset sanitization module
"""

import secrets
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DatasetSanitization:
  """
  Tool for dataset (table and images) sanitization. The sanitization includes:
    - Visualize file size distribution of the stamps
    - Drop corrupted stamps after visual inspection of the file sizes
    - Remove rows of the table without corresponding stamps

  Parameters
  ----------
  table: str, Path
    The path of the table
  images_folder: str or Path
    The path where object stamps are stored, the files must be named with the
    IAU2000 name of the object using ``iauname`` function

  See Also
  --------
  mergernet.core.utils.iauname
  """
  def __init__(
    self,
    table: Union[str, Path] = None,
    images_folder: Union[str, Path] = None
  ):
    self.table_path = Path(table)
    self.table = pd.read_csv(table)
    self.images_folder = Path(images_folder)


  def get_iauname_by_filesize(self, threshold: float) -> np.ndarray:
    """
    Filter files with file size lower than ``threshold`` and returns its
    iaunaemes

    Parameters
    ----------
    threshold: float
      The cutoff value

    Returns
    -------
    array
      Array of iaunames
    """
    folder = self.images_folder
    return np.array([
      p.stem for p in folder.glob('*')
      if p.stat().st_size < threshold * 1024
    ])


  def get_filesize_distribution(self) -> np.ndarray:
    """
    Computes the distribution of filesizes in the images folder

    Returns
    -------
    array
      The distribution of filesizes
    """
    folder = self.images_folder
    return np.array([p.stat().st_size / 1024 for p in folder.glob('*')])


  def check_images(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checks which of the objects specified in the table are or
    are not in the imeges folder

    Returns
    -------
    tuple of arrays
      A tuple of two following elements: (1) an array containing the iaunames
      of the objects **with** the corresponding image, and (2) an array
      containing iaunames of the objects **without** the corresponding image.
    """
    all_iaunames = self.table['iauname'].to_numpy()
    img_iaunames = np.array([p.stem for p in self.images_folder.glob('*')])
    mask = np.isin(all_iaunames, img_iaunames)
    return all_iaunames[mask], all_iaunames[~mask]


  def filesize_histogram(self, bins: int = 10, **kwargs):
    """
    Plot file size histogram

    Parameters
    ----------
    bins: int
      The number of bins
    kwargs: Any
      Arguments passed directly to ``plt.hist``
    """
    d = self.get_filesize_distribution()

    plt.figure(figsize=(10, 6))
    plt.hist(d, bins=bins, **kwargs)
    plt.xlabel('kB')
    plt.ylabel('Count')
    plt.show()


  def drop_images_by_iauname(self, iaunames: Union[List[str], np.ndarray]):
    """
    Remove images from ``images_folder`` by iauname

    Parameters
    ----------
    iaunames: array-like of strings
      The object iauname

    See Also
    --------
    mergernet.core.utils.iauname
    """
    files_droped = 0

    for iauname in iaunames:
      search = list(self.images_folder.glob(f'{iauname}.*'))
      if len(search) > 0:
        file = search[0]
        file.unlink()
        files_droped += 1

    print(f'Droped {files_droped} files of {len(iaunames)} found files')


  def drop_images_by_filesize(self, threshold: float):
    """
    Remove images from ``images_folder`` with file size lower than ``threshold``

    Parameters
    ----------
    threshold: float
      The cutoff value
    """
    iaunames = self.get_iauname_by_filesize(threshold)
    self.drop_images_by_iauname(iaunames)


  def sanitize(
    self,
    threshold: float = 0.0,
    dry_run: bool = False
  ) -> Union[pd.DataFrame, None]:
    """
    Sanitizes the dataset performing the follwing tasks (in this order):
      1. Remove all files with size (in kB) lower than the ``threshold``
      parameter from ``images_folder``;
      2. Remove all objects from table without corresponding file in
      ``images_folder``;
      3. Save the new table in same folder as the input table with
      ``_sanitized`` suffix added in the table name.

    If this method is called in dry-run mode, no changes will be made in the
    files, this method will just print the changes that would be made, instead

    Parameters
    ----------
    threshold: float
      The file size in kilobytes that will be dropped from the ``images_folder``
    dry_run: bool
      Set the dry-run mode if ``True``

    Returns
    -------
    DataFrame or None
      The sanitized table if not in dry-run mode
    """
    low_size_iaunames = self.get_iauname_by_filesize(threshold)
    _, no_img_iaunames = self.check_images()
    iaunames_to_exclude = np.concatenate((low_size_iaunames, no_img_iaunames))

    if dry_run:
      print('Images bellow threshold:', len(low_size_iaunames))
      print('Objects without stamps:', len(no_img_iaunames))
      print('Total files/rows to exclude:', len(iaunames_to_exclude))
    else:
      # remove images
      filtered_paths = [
        p for p in self.images_folder.glob('*')
        if p.stem in iaunames_to_exclude
      ]
      for path in filtered_paths:
        path.unlink()

      # remove rows and save table
      filtered_table = self.table[~self.table.iauname.isin(iaunames_to_exclude)]
      table_name = self.table_path.stem + '_sanitized.csv'
      save_path = self.table_path.parent / table_name
      filtered_table.to_csv(save_path, index=False)


  def sample(self, iaunames: Union[List[str], np.array]):
    """
    Creates a sample of the dataset in temp folder

    Parameters
    ----------
    iaunames: array-like of string
      The objects of the sample
    """
    tmp = Path(tempfile.gettempdir()) / f'mn_sample_{secrets.token_hex(3)}'
    tmp.mkdir(parents=True)

    filtered_files = [
      p for p in self.images_folder.glob('*')
      if p.stem in iaunames
    ]

    for file in filtered_files:
      shutil.copy(file, tmp / file.name)

    print(f'Sample created at {str(tmp)}')

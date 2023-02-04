from pathlib import Path
from typing import List, Tuple, Union

from mergernet.data.image import ImageTransform
from mergernet.services.google import GDrive
from mergernet.services.imaging import ImagingService
from mergernet.services.sciserver import SciServer


class HTTPResource:
  def __init__(self, url: str):
    self.url = url



class GoogleDriveResource:
  GD_DATASETS_PATH = 'drive/MyDrive/mergernet/datasets'

  def __init__(self, filename: Union[str, Path]):
    self.path = Path(self.GD_DATASETS_PATH) / filename



class DatasetConfig:
  """Configuration params for dataset."""
  def __init__(
    self,
    name: str = None,
    archive_url: List[Union[HTTPResource, GoogleDriveResource]] = None,
    table_url: List[str] = None,
    archive_path: Path = None,
    images_path: Path = None,
    table_path: Path = None,
    image_column: str = None,
    label_column: str = None,
    fold_column: str = 'fold',
    image_extension: str = '',
    image_shape: tuple = None,
    labels: List[str] = [],
    positions: List[Tuple[float, float]] = None,
    image_service: ImagingService = None,
    image_transform: ImageTransform = None,
    image_nested: bool = False,
  ):
    """
    Dataset Configuration Data Model

    Parameters
    ----------
    name : str, optional
      The dataset name, by default None
    archive_url : List[str], optional
      List of urls of the archive, the compressed file that contains the
      images, by default None
    table_url : List[str], optional
      List of urls of the table, the csv file that contains metadata info
      about each example, such as label, by default None
    archive_path : Path, optional
      The path where the compressed file will be stored in local file system,
      by default None
    images_path : Path, optional
      The path of the archive in local storage after descompression, this
      is used to avoid problems with different directory structures
      of the compressed images file, by default None, by default None
    table_path : Path, optional
      The path of the metadata table in local storage, by default None
    image_column : str, optional
      The name of the column in metadata table that correlates each image
      filename with respective row, usually ``iauname``, by default None
    label_column : str, optional
      The name of column that have the label of each example,
      only needed for train datasets, by default None
    fold_column : str, optional
      The name of the column that has the index of the fold the example
      belongs to, by default 'fold'
    image_extension : str, optional
      The image extension without the leading ``.``, by default ''
    image_shape : tuple, optional
      A tuple that represents the shape of each example in the following
      format: ``(HEIGHT, WIDTH, CHANNELS)``, by default None
    labels : List[str], optional
      The list of all labels, by default []
    positions : List[Tuple[float, float]], optional
      List of positions ``(RA, DEC)`` of the objects, by default None
    image_service : ImagingService, optional
      An `ImageService` instance if the dataset is not stored in cloud
      services and will be downloaded direct from the Surveys
      web-services, by default None
    image_transform : ImageTransform, optional
      An pipeline of image transformations that will be performed in
      each downloaded image, by default None
    image_nested : bool, optional
      flags if the image directory structure follows the nested pattern or
      not, by default False
    """
    self.name = name
    self.archive_url = archive_url
    self.table_url = table_url
    self.archive_path = archive_path
    self.images_path = images_path
    self.table_path = table_path
    self.image_column = image_column
    self.label_column = label_column
    self.fold_column = fold_column
    self.image_extension = image_extension
    self.image_shape = image_shape
    self.labels = labels
    self.positions = positions
    self.image_service = image_service
    self.image_transform = image_transform
    self.image_nested = image_nested
    self.n_classes = len(labels)



class DatasetRegistry:
  """
  A registry for all datasets
  """
  DARG_NO_INSPECTION = DatasetConfig(
    # table_url=GDrive.get_url('1yHnyOdXS-HKzIsbenSi646jyf2AWU9vo'),
    name='darg_no_inspection',
    archive_url=[GDrive.get_url('1ltKXhZgA4Ab60FGKCqybhebSiAL2LMgG')],
    table_url=[GDrive.get_url('1QgUYkzcjaCmo-kcgM8s2U8PYlx0C58oD')],
    archive_path=Path('sdss_lupton_jpg_128.tar.xz'),
    images_path=Path('sdss_lupton_jpg_128'),
    table_path=Path('reference_darg.csv'),
    image_column='filename',
    label_column='class',
    fold_column='fold',
    labels=['E', 'M', 'S'],
    image_shape=(128, 128, 3),
  )
  """Default configuration object for RGB dataset."""

  MESD_SDSS_128 = DatasetConfig(
    name='mesd_sdss_128',
    archive_url=[GDrive.get_url('1YZ7A9rVglf5NJp27rW8-6aJXBfBNPfYv')],
    table_url=[GDrive.get_url('1uaRXWUskBrLHi-IGLhZ5T6yIa8w5hi4H')],
    archive_path=Path('mesd_sdss_128.tar.xz'),
    images_path=Path('mesd_sdss_128'),
    table_path=Path('mesd.csv'),
    image_column='iauname',
    image_extension='jpg',
    label_column='class',
    fold_column='fold',
    labels=['merger', 'elliptical', 'spiral', 'disturbed'],
    image_shape=(128, 128, 3),
  )
  """MESD dataset with SDSS 128x128 images."""

  MESD_LEGACY_128 = DatasetConfig(
    name='mesd_legacy_128',
    archive_url=[GDrive.get_url('1cTU0SVEv3qVeVxF7pzhtOP7S-bx5cPOP')],
    table_url=[GDrive.get_url('1uaRXWUskBrLHi-IGLhZ5T6yIa8w5hi4H')],
    archive_path=Path('mesd_legacy_128.tar.xz'),
    images_path=Path('mesd_legacy_128'),
    table_path=Path('mesd.csv'),
    image_column='iauname',
    image_extension='jpg',
    label_column='class',
    fold_column='fold',
    labels=['merger', 'elliptical', 'spiral', 'disturbed'],
    image_shape=(128, 128, 3),
  )
  """MESD dataset with Legacy Survey 128x128 images."""

  BIN_SDSS_128 = DatasetConfig(
    name='bin_sdss_128',
    archive_url=[GDrive.get_url('1pZqGs6xG12Od-g3rnXEOyJ40LPAPu0J2')],
    table_url=[GDrive.get_url('1N-o1N3dYjJRaU4Nu9ZC0j2OHXzhbwYgl')],
    archive_path=Path('bin_sdss_128.tar.xz'),
    images_path=Path('bin_sdss_128'),
    table_path=Path('bin_sdss.csv'),
    image_column='iauname',
    image_extension='jpg',
    label_column='class',
    fold_column='fold',
    labels=['non_merger', 'merger'],
    image_shape=(128, 128, 3),
  )
  """Binary dataset (merger and non-merger) with SDSS 128x128 images."""

  BLIND_SPLUS_LUPTON_128 = DatasetConfig(
    name='blind_splus_lupton_128',
    archive_url=[
      SciServer.get_url('28d96731-35d9-4eac-bdee-7ccb81c5456d'),
      GDrive.get_url('1TktT_u_NTqyBPKq5KG8hW0TVYhZe6Q3r')
    ],
    table_url=[GDrive.get_url('1qd0lxMf2WzPF8bW0NciX1wwfa_mrS0Om')],
    archive_path=Path('blind_splus_lupton_128.tar.xz'),
    images_path=Path('blind_splus_lupton_128'),
    table_path=Path('blind_splus_gal80_r17_lupton.csv'),
    image_column='ID',
    image_extension='png',
    image_shape=(128, 128, 3)
  )
  """Blind dataset with S-PLUS 128x128 Lupton images."""

  BLIND_SPLUS_LUPTON_150 = DatasetConfig(
    name='blind_splus_lupton_150',
    archive_url=[
      SciServer.get_url('946b0d5c-741e-463a-bdbc-4a04313c00c7'),
      GDrive.get_url('1qERSsv9W4d2ICyKlvhhPaR7a2j6W5HrD')
    ],
    table_url=GDrive.get_url('1qd0lxMf2WzPF8bW0NciX1wwfa_mrS0Om'),
    archive_path=Path('blind_splus_lupton_150.tar.xz'),
    images_path=Path('blind_splus_lupton_150'),
    table_path=Path('blind_splus_gal80_r17_lupton.csv'),
    image_column='ID',
    image_extension='png',
    image_shape=(150, 150, 3)
  )
  """Blind dataset with S-PLUS 150x150 Lupton images."""

  BLIND_SPLUS_TRILOGY_128 = DatasetConfig(
    name='blind_splus_trilogy_128',
    archive_url=[
      SciServer.get_url('4fe38e2a-c1db-4cd3-92ec-1c2c5b2c5284'),
      GDrive.get_url('1lPvOtE6HCJ7Xi5hAGfjgvrfwGiLRT7uH')
    ],
    table_url=[GDrive.get_url('1JEjQleflgQf_L0Qkun15PHUn-OnOG0rG')],
    archive_path=Path('blind_splus_trilogy_128.tar.xz'),
    images_path=Path('blind_splus_trilogy_128'),
    table_path=Path('blind_splus_gal80_r17_trilogy.csv'),
    image_column='ID',
    image_extension='png',
    image_shape=(128, 128, 3)
  )
  """Blind dataset with S-PLUS 128x128 Trilogy images."""

  BLIND_SPLUS_TRILOGY_150 = DatasetConfig(
    name='blind_splus_trilogy_150',
    archive_url=[
      SciServer.get_url('74ca0cba-6c9e-4022-95b8-9d7964d70947'),
      GDrive.get_url('1aiLtmbrJSRQTmSB3XP1AqXi0tfUi3S-H')
    ],
    table_url=[GDrive.get_url('1JEjQleflgQf_L0Qkun15PHUn-OnOG0rG')],
    archive_path=Path('blind_splus_trilogy_150.tar.xz'),
    images_path=Path('blind_splus_trilogy_150'),
    table_path=Path('blind_splus_gal80_r17_trilogy.csv'),
    image_column='ID',
    image_extension='png',
    image_shape=(150, 150, 3)
  )
  """Blind dataset with S-PLUS 150x150 Trilogy images."""

  BIN_LEGACY_NORTH_RGB_128 = DatasetConfig(
    name='bin_legacy_north_rgb_128',
    archive_url=[
      SciServer.get_url('9e1fd5f9-64b6-4130-a409-b7dde8f65c0e'),
      GDrive.get_url('1lblw2-Bvqs6q4T6fjVmc1kKm54Ps6rMs')
    ],
    table_url=[GDrive.get_url('1-kvD6eEuqybNwu-Cds_p5FnnrqAsBL_1')],
    archive_path=Path('bin_legacy_north_rgb_128.tar.xz'),
    images_path=Path('bin_legacy_north_rgb_128'),
    table_path=Path('bin_legacy_north_sanitized.csv'),
    image_column='iauname',
    image_extension='jpg',
    label_column='class',
    fold_column='fold',
    labels=['non_merger', 'merger'],
    image_shape=(128, 128, 3),
  )
  """
  Binary dataset (merger and non-merger) with Legacy 128x128 RGB images
  on north sky with of pixscale 0.55 arcsec/pixel.
  """

  BLIND_SPLUS_GAL80_LS10_RGB_128 = DatasetConfig(
    name='BLIND_SPLUS_GAL80_LS10_RGB_128',
    archive_url=[
      SciServer.get_url('bd9c6c3c-88ae-4b60-a255-6e47b77d8c2d')
    ],
    table_url=[SciServer.get_url('b255ba16-5f3c-4d8a-8ade-e268c65692d4')],
    archive_path=Path('blind_splus_gal80_r13.5-17_ls10_rgb_128.tar.xz'),
    images_path=Path('blind_splus_gal80_r13.5-17_ls10_rgb_128'),
    table_path=Path('blind_splus_gal80_r13.5-17_sanitized.csv'),
    image_column='iauname',
    image_extension='jpg',
    image_shape=(128, 128, 3)
  )
  """
  S-PLUS blind sample with 13.5 < r_auto < 17 with ls-10-early-grz RGB images
  128x128 pixscale of with 0.55 arcsec/pixel
  """

  DECALS_0364_1M_PART0 = DatasetConfig(
    name='DECALS_0364_1M_PART0',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part0.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part0.tar.xz'),
    images_path=Path('decals_0.364_png_part0'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART1 = DatasetConfig(
    name='DECALS_0364_1M_PART1',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part1.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part1.tar.xz'),
    images_path=Path('decals_0.364_png_part1'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART2 = DatasetConfig(
    name='DECALS_0364_1M_PART2',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part2.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part2.tar.xz'),
    images_path=Path('decals_0.364_png_part2'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART3 = DatasetConfig(
    name='DECALS_0364_1M_PART3',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part3.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part3.tar.xz'),
    images_path=Path('decals_0.364_png_part3'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART4 = DatasetConfig(
    name='DECALS_0364_1M_PART4',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part4.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part4.tar.xz'),
    images_path=Path('decals_0.364_png_part4'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART5 = DatasetConfig(
    name='DECALS_0364_1M_PART5',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part5.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part5.tar.xz'),
    images_path=Path('decals_0.364_png_part5'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART6 = DatasetConfig(
    name='DECALS_0364_1M_PART6',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part6.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part6.tar.xz'),
    images_path=Path('decals_0.364_png_part6'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART7 = DatasetConfig(
    name='DECALS_0364_1M_PART7',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part7.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part7.tar.xz'),
    images_path=Path('decals_0.364_png_part7'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART8 = DatasetConfig(
    name='DECALS_0364_1M_PART8',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part8.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part8.tar.xz'),
    images_path=Path('decals_0.364_png_part8'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART9 = DatasetConfig(
    name='DECALS_0364_1M_PART9',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part9.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part9.tar.xz'),
    images_path=Path('decals_0.364_png_part9'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART10 = DatasetConfig(
    name='DECALS_0364_1M_PART10',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part10.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part10.tar.xz'),
    images_path=Path('decals_0.364_png_part10'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART11 = DatasetConfig(
    name='DECALS_0364_1M_PART11',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part11.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part11.tar.xz'),
    images_path=Path('decals_0.364_png_part11'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )

  DECALS_0364_1M_PART12 = DatasetConfig(
    name='DECALS_0364_1M_PART12',
    archive_url=[
      GoogleDriveResource('decals_0.364_png_part12.tar.xz'),
    ],
    archive_path=Path('decals_0.364_png_part12.tar.xz'),
    images_path=Path('decals_0.364_png_part12'),
    image_extension='png',
    image_shape=(224, 224, 3),
    image_nested=True,
  )


_DECALS_0364_1M_DOC = """
  DECALS 1M dataset with colored png images PART {}

  .. list-table:: Dataset parameters
    :header-rows: 1

    * - Parameter
      - Value
    * - Type
      - Predictions
    * - Pix scale
      - 0.364
    * - Shape
      - 244,244,3
    * - Image Format
      - PNG
    * - Bands
      - GRZ
    * - Scale G
      - (2, 0.008)
    * - Scale R
      - (1, 0.014)
    * - Scale Z
      - (0, 0.019)
    * - MinMax
      - (-0.5, 300)
    * - Brightness
      - 1.3
    * - Desaturate
      - True
    * - Non-Linearity
      - asinh2
"""

DatasetRegistry.DECALS_0364_1M_PART0.__doc__ = _DECALS_0364_1M_DOC.format(0)
DatasetRegistry.DECALS_0364_1M_PART1.__doc__ = _DECALS_0364_1M_DOC.format(1)
DatasetRegistry.DECALS_0364_1M_PART2.__doc__ = _DECALS_0364_1M_DOC.format(2)
DatasetRegistry.DECALS_0364_1M_PART3.__doc__ = _DECALS_0364_1M_DOC.format(3)
DatasetRegistry.DECALS_0364_1M_PART4.__doc__ = _DECALS_0364_1M_DOC.format(4)
DatasetRegistry.DECALS_0364_1M_PART5.__doc__ = _DECALS_0364_1M_DOC.format(5)
DatasetRegistry.DECALS_0364_1M_PART6.__doc__ = _DECALS_0364_1M_DOC.format(6)
DatasetRegistry.DECALS_0364_1M_PART7.__doc__ = _DECALS_0364_1M_DOC.format(7)
DatasetRegistry.DECALS_0364_1M_PART8.__doc__ = _DECALS_0364_1M_DOC.format(8)
DatasetRegistry.DECALS_0364_1M_PART9.__doc__ = _DECALS_0364_1M_DOC.format(9)
DatasetRegistry.DECALS_0364_1M_PART10.__doc__ = _DECALS_0364_1M_DOC.format(10)
DatasetRegistry.DECALS_0364_1M_PART11.__doc__ = _DECALS_0364_1M_DOC.format(11)
DatasetRegistry.DECALS_0364_1M_PART12.__doc__ = _DECALS_0364_1M_DOC.format(12)

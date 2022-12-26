from pathlib import Path
from typing import List, Tuple

from mergernet.data.image import ImageTransform
from mergernet.services.google import GDrive
from mergernet.services.imaging import ImagingService
from mergernet.services.sciserver import SciServer


class DatasetConfig:
  """Configuration params for dataset."""
  def __init__(
    self,
    name: str = None,
    archive_url: List[str] = None,
    table_url: List[str] = None,
    archive_path: Path = None,
    images_path: Path = None,
    table_path: Path = None,
    image_column: str = None,
    label_column: str = None,
    fold_column: str = 'fold',
    image_extension: str = '',
    image_shape: tuple = None,
    labels: dict = [],
    positions: List[Tuple[float, float]] = None,
    image_service: ImagingService = None,
    image_transform: ImageTransform = None,
    image_nested: bool = False,
  ):
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

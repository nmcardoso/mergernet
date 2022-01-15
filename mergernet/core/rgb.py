from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from PIL import Image
import numpy as np

from mergernet.core.trilogy import MakeImg



class RGB:
  def trilogy_fits_to_png(
    self,
    r_fits: Path,
    g_fits: Path,
    b_fits: Path,
    ra: float,
    dec: float,
    save_path: Path,
    sky_size: float = 70.4,
    fig_size: int = 128
  ):
    r_hdu = fits.open(r_fits)
    g_hdu = fits.open(g_fits)
    b_hdu = fits.open(b_fits)

    position = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs', equinox='J2000.0')
    size = u.Quantity((sky_size, sky_size), u.arcsec)
    r_wcs = WCS(r_hdu[0].header)
    g_wcs = WCS(g_hdu[0].header)
    b_wcs = WCS(b_hdu[0].header)

    r_cutout = Cutout2D(r_hdu[0].data, position, size, wcs=r_wcs)
    g_cutout = Cutout2D(g_hdu[0].data, position, size, wcs=g_wcs)
    b_cutout = Cutout2D(b_hdu[0].data, position, size, wcs=b_wcs)

    r = r_cutout.data
    g = g_cutout.data
    b = b_cutout.data

    rgb = np.dstack((r, g, b))

    img = Image.fromarray(rgb)

    if img.size[0] != img.size[1] != fig_size:
      img = img.resize((fig_size, fig_size))

    img.save(save_path)


  def make_trilogy_fits(
    self,
    source_dict: dict,
    r_path: Path,
    g_path: Path,
    b_path: Path,
    type: str = 'sdss', # 'sdss' or 'splus'
  ):
    pass

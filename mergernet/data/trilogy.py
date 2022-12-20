"""
Trilogy Program

All credits to original author Dan Coe (https://www.stsci.edu/~dcoe/trilogy/Intro.html)
with modifications by Gustavo B. O. Schwarz (https://github.com/schwarzam/)
"""


from os import listdir
from os.path import isfile, join

import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import golden

rw, gw, bw = 0.299,  0.587,  0.114  # NTSC (also used by PIL in "convert")
rw, gw, bw = 0.3086, 0.6094, 0.0820  # linear
rw, gw, bw = 0.212671, 0.715160, 0.072169  # D65: red boosted, blue muted a bit, I like it



def get_levels(stampsRGB, unsatpercent):
  datasorted = np.sort(stampsRGB.flat)
  datasorted[np.isnan(datasorted)] = 0

  noisesig = 1

  rem, m, r = meanstd(datasorted)
  x0 = 0

  x1 = m + noisesig * r

  x2 = setLevel(datasorted, unsatpercent)

  levels = x0, x1, x2

  return levels



def get_clip(m, m_min=None, m_max=None):
  if m_min == None:
    m_min = min(m)
  if m_max == None:
    m_max = max(m)

  return np.clip(m, m_min, m_max)



def setLevel(data, pp):
  vs = data
  vs = get_clip(vs, 0, None)

  ii = np.array(pp) * len(vs)
  ii = ii.astype('int')
  ii = np.clip(ii, 0, len(vs) - 1)
  levels = vs.take(ii)
  return levels



def rms(x):
  return np.sqrt(np.mean(x ** 2))



def meanstd(datasorted, n_sigma = 3, n = 5):
  x = datasorted

  ihi = nx = len(x)
  ilo = 0

  xsort = x

  for i in range(n):
    xs = xsort[ilo: ihi]

    imed = (ilo+ihi) / 2

    aver = xs[int(imed)]

    std1 = np.std(xs)
    std1 = rms(xs - aver)

    lo = aver - n_sigma * std1
    hi = aver + n_sigma * std1

    ilo = np.searchsorted(xsort, lo)
    ihi = np.searchsorted(xsort, hi, side='right')

    nnx = ihi - ilo

    if nnx == nx:
      break
    else:
      nx = nnx

  remaining = xrem = xs[ilo:ihi]
  mean = np.mean(xrem)
  std = rms(xrem - mean)

  return remaining, mean, std



def RGB2im(RGB):
  data = RGB
  data = np.transpose(data, (1,2,0))  # (3, ny, nx) -> (ny, nx, 3)
  data = np.clip(data, 0, 255)
  data = data.astype('uint8')
  three = data.shape[-1]  # 3 if RGB, 1 if L
  if three == 3:
    im = Image.fromarray(data)
  elif three == 1:
    im = Image.fromarray(data[:,:,0], 'L')
  else:
    print('Data shape not understood: expect last number to be 3 for RGB, 1 for L', data.shape)
    raise Exception  # Raise generic exception and exit

  im = im.transpose(Image.FLIP_TOP_BOTTOM)
  return im



def da(k):
  a1 = k * (x1 - x0) + 1
  a2 = k * (x2 - x0) + 1
  a1n = a1 ** n
  a1n = np.abs(a1n)

  da1 = a1n - a2
  k = np.abs(k)

  if k == 0:
    return da(1e-10)
  else:
    da1 = da1 / k

  return np.abs(da1)



def imscale(data, levels, y1):
  global n, x0, x1, x2
  x0, x1, x2 = levels
  if y1 == 0.5:
    k = (x2 - 2 * x1 + x0) / float(x1 - x0) ** 2
  else:
    n = 1 / y1
    k = abs(golden(da))

  r1 = np.log10(k* (x2-x0) + 1)

  v = np.ravel(data)
  v = get_clip(v, 0, None)

  d = k * (v - x0) + 1
  d = get_clip(d, 1e-30, None)

  z = np.log10(d) / r1
  z = np.clip(z, 0, 1)
  z.shape = data.shape

  z = z * 255
  z = z.astype('uint8')

  return z



def satK2m(K):
  m00 = rw * (1-K) + K
  m01 = gw * (1-K)
  m02 = bw * (1-K)

  m10 = rw * (1-K)
  m11 = gw * (1-K) + K
  m12 = bw * (1-K)

  m20 = rw * (1-K)
  m21 = gw * (1-K)
  m22 = bw * (1-K) + K

  m = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
  return m



def adjust_saturation(RGB, K):
  m = satK2m(K)
  three, nx, ny = RGB.shape

  RGB.shape = three, nx*ny

  RGB = np.dot(m, RGB)
  RGB.shape = three, nx, ny
  return RGB



class MakeImg():
  def __init__(
    self,
    pathdict=None,
    pathstack=None,
    arraydict=None,
    noiselum = 0.15,
    satpercent = 0.15,
    colorsatfac = 2
  ):
    pass
    # sgn = 1

    # unsatpercent = 1 - 0.01 * satpercent

    # noiselums = {'R': noiselum, 'G': noiselum, 'B': noiselum}

    # dx = arraydict['R'][0].shape[0]
    # dy = arraydict['R'][0].shape[0]

    # stamp = np.zeros((3, int(dy), int(dx)))


    # _, nx, ny = stamp.shape

    # for i, channel in enumerate('RGB'):
    #   noiselum = noiselums[channel]

    #   for image in arraydict[channel]:
    #     stamp[i] += sgn * image

    #   level = get_levels(stamp[i], unsatpercent)
    #   stamp[i] = imscale(stamp[i], level, noiselum)

    # stamp = adjust_saturation(stamp, colorsatfac)
    # self.im = RGB2im(stamp)


  def color(
    self,
    image: np.ndarray,
    noiselum: float = 0.15,
    satpercent: float = 0.15,
    colorsatfac: float = 2,
    inplace: bool = True
  ):
    unsatpercent = 1 - 0.01 * satpercent

    noiselums = {'R': noiselum, 'G': noiselum, 'B': noiselum}

    if inplace:
      stamp = image
    else:
      stamp = np.copy(image)

    for i, channel in enumerate('RGB'):
      noiselum = noiselums[channel]
      level = get_levels(stamp[i], unsatpercent)
      stamp[i] = imscale(stamp[i], level, noiselum)

    stamp = adjust_saturation(stamp, colorsatfac)
    self.im = RGB2im(stamp)


  def savefig(self, path):
    plt.imsave(arr = np.array(self.im), fname = path)


  def get_array(self):
    return np.array(self.im)

if __name__ == '__main__':
  image = []
  r = ['R', 'I', 'F861', 'Z']
  g = ['F660']
  b = ['F378', 'F395', 'F410', 'F430']

  for band in [r, g, b]:
    data = np.zeros((11000, 11000))
    for b in band:
      data += astropy.io.fits.getdata(f'../../data/s82/{b}.fz')
    image.append(data)

  image = np.array(image)

  m = MakeImg()
  m.color(image)
  m.savefig('test.png')

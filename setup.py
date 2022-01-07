from setuptools import setup, find_packages

setup(
  name='mergernet',
  version='0.1.0',
  description='deep learning model',
  author='Natanael',
  author_email='nauxmac@gmail.com',
  packages=find_packages(),
  install_requires=[
    'wheel',
    'numpy>=1.19.5',
  ]
)

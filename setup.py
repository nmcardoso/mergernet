from setuptools import find_packages, setup

setup(
  name='mergernet',
  version='0.2.0',
  description='deep learning model',
  author='Natanael',
  author_email='nauxmac@gmail.com',
  packages=find_packages(),
  include_package_data=True,
  package_data={
    'mergernet': []
  },
  install_requires=[
    'wheel',
    'numpy>=1.21.6',
    'pandas>=1.3.5',
    'tensorflow>=2.9.2',
    'scikit-learn>=1.0.2',
    'tqdm>=4.64.1',
    'requests>=2.23',
    'Pillow>=7.1.2',
    'astropy>=5.2',
    'tensorflow_addons',
    'tensorboard',
    'optuna',
    'plotly',
    'wandb',
    'zoobot==0.0.4',
    'keras_applications', # zoobot
    'tensorflow_probability>=0.11', # zoobot
    'seaborn' , # zoobot
    'pydot>=1.4.2', # tf.keras.utils.plot_model
  ],
  extras_require={
    'docs': [
      'Jinja2>=3.1',
      'sphinxcontrib-napoleon',
      'sphinx',
      'furo',
      'ipykernel',
      'sphinx_copybutton',
    ]
  }
)

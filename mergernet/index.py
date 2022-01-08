# https://www.geeksforgeeks.org/absolute-and-relative-imports-in-python/
# https://docs.python-guide.org/writing/structure/

# python -m pip install 'git+https://github.com/myorg/myrepo.git#egg=python_tools&subdirectory=python-tools'
# https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support
# https://packaging.python.org/en/latest/overview/
# https://packaging.python.org/en/latest/tutorials/packaging-projects/

import argparse

def legacy(args):
  print('cuceta')

# Main program
parser = argparse.ArgumentParser(prog='mn')
subparsers = parser.add_subparsers(title='Modules')

# Module: download
download_parser = subparsers.add_parser('download', help='data acquisition module')
download_subparsers = download_parser.add_subparsers()

# Module: downlods/legacy
legacy_parser = download_subparsers.add_parser('legacy', help='Legacy Survey Downloads')
legacy_parser.add_argument('--rgb', action='store_true', help='Download legacy RGB images')
legacy_parser.add_argument('--csv', action='store', help='Download images listed in a CSV file with following columns: RA, DEC and PATH (the path where image will be downloaded)')
legacy_parser.set_defaults(func=legacy)


args = parser.parse_args(['download', 'legacy', '--rgb'])
print(args)
args.func(args)

if __name__ == '__main__':
  parser.parse_args(['-h'])

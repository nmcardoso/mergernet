import importlib
import sys
import os

# sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
# mod = __import__('..core.logger', globals(), locals(), [], 1)
# print(mod)

m = importlib.import_module('..core.logger', package='mergernet.core')
print(dir(m))
Logger = m.Logger
print(Logger)

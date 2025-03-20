"""FFI Package for MyLib"""
import os
import sys

from . import base
from . import function
from . import registry

lib_path = os.path.dirname(base._get_lib_path()[0])
if lib_path not in sys.path:
    sys.path.append(lib_path)

try:
    import libmylib
except ImportError:
    sys.stderr.write("warning: Failed to import libmylib. Make sure it's in your Python path.\n")

get_global_func = function.get_global_func
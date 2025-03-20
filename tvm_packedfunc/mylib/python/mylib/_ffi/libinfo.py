"""Library information"""
import os
import sys
import platform

def find_lib_path():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

    lib_search_paths = [
        os.path.join(curr_path, "../../build/lib"),
        os.path.join(curr_path, "../../lib"),
        os.path.join(sys.prefix, "lib"),
        "/usr/local/lib",
        "/usr/lib"
    ]

    if sys.platform.startswith("win32"):
        lib_name = "mylib.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "libmylib.dylib"
    else:
        lib_name = "libmylib.so"

    lib_paths = [os.path.join(path, lib_name) for path in lib_search_paths]

    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            return [lib_path]
        
    raise RuntimeError(
        "Cannot find the MyLib library. " +
        "List of candidates:\n" + "\n".join(lib_paths)
    )
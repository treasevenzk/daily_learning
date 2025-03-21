"""Setup script for PackedFun."""

import os
import platform
from setuptools import setup, find_packages

# Try to find the PackedFun library
def find_lib_path():
    """Find PackedFun library."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # Search in various possible locations
    lib_paths = [
        os.path.join(curr_path, '../build/lib/'),
        os.path.join(curr_path, '../build/'),
        os.path.join(curr_path, '../lib/'),
    ]
    
    lib_name = 'libpackedfun'
    if platform.system() == 'Windows':
        lib_name = 'packedfun.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'libpackedfun.dylib'
    else:
        lib_name = 'libpackedfun.so'
    
    lib_paths = [os.path.join(p, lib_name) for p in lib_paths]
    lib_paths = [p for p in lib_paths if os.path.exists(p) and os.path.isfile(p)]
    
    if not lib_paths:
        raise RuntimeError('Cannot find the PackedFun library. Make sure it is built.')
    
    return lib_paths[0]

# Copy the library to the package directory
if not os.path.exists('packedfun/_lib'):
    os.makedirs('packedfun/_lib')

lib_path = find_lib_path()
target_path = os.path.join('packedfun/_lib', os.path.basename(lib_path))
if not os.path.exists(target_path):
    import shutil
    shutil.copy(lib_path, target_path)

setup(
    name='packedfun',
    version='0.1.0',
    description='C++ function binding for Python',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    package_data={
        'packedfun': ['_lib/*'],
    },
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
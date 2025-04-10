"""Setup script for MyLib package."""
import os
import platform
from setuptools import setup, find_packages

current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_dir, ".."))

setup(
    name="mylib",
    version="0.1.0",
    description="MyLib Python package with C++ backend",
    author="User",
    author_email="user@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Lincense :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6"
)
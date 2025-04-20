#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools
from pathlib import Path

NAME = "learnpdes"
EXCLUDE_DIRS = ["tests"]
VERSION = "1.0"
AUTHOR = "Leon Deligny"
URL = "https://github.com/cfsengineering/CEASIOMpy"
REQUIRES_PYTHON = ">=3.12.0"
README = "README.md"
PACKAGE_DIR = "."

here = Path(__file__).parent

with open(Path(here, README), "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    long_description=long_description,
    url=URL,
    include_package_data=True,
    package_dir={"": PACKAGE_DIR},
    packages=setuptools.find_packages(exclude=EXCLUDE_DIRS),
    python_requires=REQUIRES_PYTHON,
    install_requires=[],
    # See: https://pypi.org/classifiers/
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
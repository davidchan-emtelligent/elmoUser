#!/usr/bin/python
from setuptools import setup, find_packages

setup(
    name='bilm',
    version='0.1',
    url='http://github.com/allenai/bilm-tf',
    packages=find_packages(),
    tests_require=[],
    zip_safe=False,
    entry_points='',
    install_requires=[
        'typing',
        'tensorflow-gpu==1.2',
        'h5py',
        ],
)


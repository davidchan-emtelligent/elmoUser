#!/usr/bin/python
from setuptools import setup, find_packages

setup(
    name='elmoUser',
    version='0.1',
    python_requires='>=3.4',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    entry_points={
        'console_scripts': [
            'trainer=elmoUser.trainer:main',
        ],
    },
    package_data={
	'elmoUser': ['resources/*'],
    },

    install_requires=[
        'bilm',
        ],
)


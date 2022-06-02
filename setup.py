#!/usr/bin/enc python

from setuptools import setup, find_packages

setup(
    name = "past",
    version = "0.0.1",
    keywords = ("pip", "past"),
    description = "PAST: a Prior-based self-Attention method for Spatial Transcriptomics",
    long_description = "PAST: a Prior-based self-Attention method for Spatial Transcriptomics",
    license = "MIT License",
    url = "https://github.com/lizhen18THU/PAST",
    author = "Zhen Li",
    author_email = "lizhen18THU@163.com",
    packages = find_packages(),
    python_requires = ">3.6.0",
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=[
        'numba',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'scanpy',
        'torch',
    ]
)
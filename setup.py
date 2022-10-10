#!/usr/bin/enc python

from setuptools import setup, find_packages

setup(
    name = "bio-past",
    version = "1.1.0",
    keywords = ("pip", "past"),
    description = "PAST: latent feature extraction with a Prior-based self-Attention framework for Spatial Transcriptomics",
    long_description = "PAST: latent feature extraction with a Prior-based self-Attention framework for Spatial Transcriptomics. Recent development of spatial transcriptomics (ST), which can not only obtain comprehensive gene expression profiles but also preserve spatial information, provides a new dimension to genomics research. As a prerequisite and basis for various downstream missions, latent feature extraction is of great significance for the analysis of spatial transcriptomics. However, few methods consider facilitating data analysis via integrating rich prior information from existing data, and the modality fusion of spatial information and gene expression also remains challenging. Here we propose PAST, a representation learning framework for spatial transcriptomics which takes advantage of prior information with Bayesian neural network and integrates spatial information and gene profile with self-attention mechanism",
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
        'numba>=0.55.1',
        'numpy>=1.21.6',
        'pandas>=1.3.4',
        'scipy>=1.8.1',
        'scikit-learn>=1.1.1',
        'scanpy>=1.9.1',
        'torch>=1.9.1',
        'rpy2>=3.4.5',
    ]
)

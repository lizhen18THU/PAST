from setuptools import setup, find_packages

setup(
    name = "bio-past",
    version = "1.6.3",
    keywords = ["pip", "past"],
    description = "PAST: latent feature extraction with a Prior-based self-Attention framework for Spatial Transcriptomics",
    long_description = "PAST software is build on a variational graph convolutional auto-encoder designed for spatial transcriptomics which integrates prior information with Bayesian neural network, captures spatial information with self-attention mechanism and enables scalable application with ripple walk sampler strategy. PAST could effectively characterize spatial domains and facilitate various downstream analysis through integrating spatial information and reference from various sources. Besides, PAST also enable time and memory-efficient application on large datasets while preserving global spatial patterns for better performance. Importantly, PAST could also facilitate accurate annotation of spatial domains and thus provide biological insights.",
    license = "MIT License",
    url = "https://github.com/lizhen18THU/PAST",
    author = "Zhen Li",
    author_email = "lizhen18@tsinghua.org.cn",
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
        'numpy>=1.21.5',
        'pandas>=1.3.4',
        'scipy>=1.5.3',
        'scikit-learn>=1.0.2',
        'scanpy>=1.9.1',
        'torch>=1.9.1',
    ]
)

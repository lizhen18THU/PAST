[![PyPI](https://img.shields.io/pypi/v/bio-past.svg)](https://pypi.org/project/bio-past)
[![Downloads](https://pepy.tech/badge/bio-past)](https://pepy.tech/project/bio-past)
[![Documentation Status](https://readthedocs.org/projects/past/badge/?version=latest)](https://past.readthedocs.io/en/latest/)
# PAST: a Prior-based self-Attention method for Spatial Transcriptomics

<div align=center>
<img src = "docs/source/PAST_LOGO.png" width = 65% height = 65%>
</div>

### Find more details and tutorials on [the Documentation of PAST.](https://past.readthedocs.io/en/latest/)

## Overview
PAST software is build on a variational graph convolutional auto-encoder designed for spatial transcriptomics which integrates prior information with Bayesian neural network, captures spatial information with self-attention mechanism and enables scalable application with ripple walk sampler strategy. PAST could effectively characterize spatial domains and facilitate various downstream analysis through integrating spatial information and reference from various sources. Besides, PAST also enable time and memory-efficient application on large datasets while preserving global spatial patterns for better performance. Importantly, PAST could also facilitate accurate annotation of spatial domains and thus provide biological insights.

<div align=center>
<img src = "docs/source/PAST_Overview.png" width = 100% height = 100%>
</div>

## Getting Started
### Installation
PAST is available on PyPI [here](https://pypi.org/project/bio-past) and can be installed via
```
pip install bio-past
```

You can also install PAST from GitHub via
```
git clone https://github.com/lizhen18THU/PAST.git
cd PAST
python setup.py install
```

### Dependency
numba   
numpy   
pandas   
scipy   
scikit-learn   
scanpy   
torch   

These dependencies will be automatically installed along with PAST. To implement the mclust algorithm with python, the rpy2 package and the mclust package is needed. See [rpy2](https://pypi.org/project/rpy2) and [mclust](https://cran.r-project.org/web/packages/mclust/index.html) for detail.

## Citation

[![PyPI](https://img.shields.io/pypi/v/bio-past.svg)](https://pypi.org/project/bio-past)
[![Downloads](https://pepy.tech/badge/bio-past)](https://pepy.tech/project/bio-past)
[![Documentation Status](https://readthedocs.org/projects/past/badge/?version=latest)](https://past.readthedocs.io/en/latest/)
# PAST: a Prior-based self-Attention method for Spatial Transcriptomics

<div align=center>
<img src = "docs/source/PAST_LOGO.png" width = 65% height = 65%>
</div>

### Find more details and tutorials on [the Documentation of PAST](https://past.readthedocs.io/en/latest/).

## Overview
Rapid advances in spatial transcriptomics (ST) have revolutionized the interrogation of spatial heterogeneity and increase the demand for comprehensive methods to effectively characterize spatial domains. As a prerequisite for ST data analysis, spatial domain characterization is a crucial step for downstream analyses and biological implications. Here we propose PAST, a variational graph convolutional auto-encoder for ST, which effectively integrates prior information via a Bayesian neural network, captures spatial patterns via a self-attention mechanism, and enables scalable application via a ripple walk sampler strategy. Through comprehensive experiments on datasets generated by different technologies, we demonstrated that PAST could effectively characterize spatial domains and facilitate various downstream analyses, including ST visualization, spatial trajectory inference and pseudo-time analysis. Besides, we also highlight the advantages of PAST for multi-slice joint embedding and automatic annotation of spatial domains in newly sequenced ST data. Compared with existing methods, PAST is featured as the first ST method which integrates reference data to analyze ST data and we anticipate that PAST will open up new horizons for researchers to decipher ST data with customized reference data, which dramatically expands the applicability of ST technology.

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
```
numba   
numpy   
pandas   
scipy   
scikit-learn   
scanpy   
torch
```

These dependencies will be automatically installed along with PAST. To implement the mclust algorithm with python, the rpy2 package and the mclust package is needed. See [rpy2](https://pypi.org/project/rpy2) and [mclust](https://cran.r-project.org/web/packages/mclust/index.html) for detail.

## Citation

Li, Z., Chen, X., Zhang, X., Chen, S., & Jiang, R. (2022). PAST: latent feature extraction with a Prior-based self-Attention framework for Spatial Transcriptomics. bioRxiv, 2022.11.09.515447. doi:10.1101/2022.11.09.515447


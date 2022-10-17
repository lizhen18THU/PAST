Installation
====

PyPI
----

PAST is available on PyPI here_ and can be installed via::

    pip install bio-past

GitHub
----

PAST can also installed from GitHub via::

    git clone https://github.com/lizhen18THU/PAST.git
    cd PAST
    python setup.py install

Dependency
----
::
    numba
    numpy
    pandas
    scipy
    scikit-learn
    scanpy
    torch

These dependencies will be automatically installed along with PAST. To implement the mclust algorithm with python, the rpy2 package and the mclust package is needed. See rpy2_ and mclust_ for detail.

.. _here: https://pypi.org/project/bio-past
.. _rpy2: https://pypi.org/project/rpy2
.. _mclust: https://cran.r-project.org/web/packages/mclust/index.html

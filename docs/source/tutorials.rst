Tutorials
=========

.. |colab-getting-started| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/1%20-%20Getting%20Started.ipynb

.. |colab-vector| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/2%20-%20Getting%20Started%20with%20Vector%20Data.ipynb

.. |colab-advanced-fitting| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/3%20-%20Advanced%20Function%20Fitting.ipynb

.. |colab-nonpoly-fitting| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/4%20-%20Fitting%20non-polynomial%20functions.ipynb

.. |colab-exporting| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/5%20-%20Exporting%20Data.ipynb

.. |colab-synthetic| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/6%20-%20Recovering%20SDEs%20from%20simulated%20time%20series.ipynb

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg

The following tutorial notebooks will familiarize you with the various functionalities of PyDaddy. These notebooks can be executed on Google Colab by clicking the |colab| buttons (no installation required). You can also download these notebooks from the `GitHub repo. <https://github.com/tee-lab/PyDaddy/tree/master/notebooks>`_

Getting started with PyDaddy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-getting-started| This notebook introduces the basic functionalities of PyDaddy, using a 1-dimensional dataset. The notebook explores how to visually inspect drift and diffusion functions, how to fit analytical expressions to them, and how to use the various diagnostic tools provided.

Getting started with vector data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-vector| PyDaddy also works with 2-dimensional vector data. This notebook demonstrates PyDaddy operation with a vector dataset.

Advanced function fitting
^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-advanced-fitting| PyDaddy can discover analytical expressions for the drift and diffusion functions. This notebook describes how to customize the fitting procedure to obtain best results.

Fitting non-polynomial functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-nonpoly-fitting| By default, PyDaddy fits polynomials for the drift and diffusion functions. However, this behaviour can be customized by providing a custom library of functions for the sparse regression procedure, this notebook demonstrates how to do this.

Exporting data
^^^^^^^^^^^^^^
|colab-exporting| This notebook demonstrates how to export the recovered drift and diffusion data as CSV files or Pandas data-frames.

Recovering SDEs from synthetic time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-synthetic| This notebook generates a simulated time series from a user-specified SDE, and uses PyDaddy to recover the drift and diffusion functions from the simulated time series.
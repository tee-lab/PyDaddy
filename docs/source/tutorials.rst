Tutorials
=========

.. |colab-getting-started| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/1%20-%20Getting%20Started.ipynb

.. |github-getting-started| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github
    :target: https://www.github.com/tee-lab/PyDaddy/blob/master/notebooks/1%20-%20Getting%20Started.ipynb

.. |colab-vector| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/2%20-%20Getting%20Started%20with%20Vector%20Data.ipynb

.. |github-vector| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github
    :target: https://www.github.com/tee-lab/PyDaddy/blob/master/notebooks/2%20-%20Getting%20Started%20with%20Vector%20Data.ipynb

.. |colab-advanced-fitting| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/3%20-%20Advanced%20Function%20Fitting.ipynb

.. |github-advanced-fitting| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github
    :target: https://www.github.com/tee-lab/PyDaddy/blob/master/notebooks/3%20-%20Advanced%20Function%20Fitting.ipynb

.. |colab-nonpoly-fitting| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/4%20-%20Fitting%20non-polynomial%20functions.ipynb

.. |github-nonpoly-fitting| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github
    :target: https://www.github.com/tee-lab/PyDaddy/blob/master/notebooks/4%20-%20Fitting%20non-polynomial%20functions.ipynb

.. |colab-exporting| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/5%20-%20Exporting%20Data.ipynb

.. |github-exporting| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github
    :target: https://www.github.com/tee-lab/PyDaddy/blob/master/notebooks/5%20-%20Exporting%20Data.ipynb

.. |colab-synthetic| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/tee-lab/PyDaddy/blob/master/notebooks/6%20-%20Recovering%20SDEs%20from%20simulated%20time%20series.ipynb

.. |github-synthetic| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github
    :target: https://www.github.com/tee-lab/PyDaddy/blob/master/notebooks/6%20-%20Recovering%20SDEs%20from%20simulated%20time%20series.ipynb

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg

.. |github| image:: https://img.shields.io/badge/Open_in_GitHub-grey?logo=github

The following tutorial notebooks will familiarize you with the various functionalities of PyDaddy. These notebooks can be executed on Google Colab by clicking the |colab| buttons (no installation required).

.. note::

    While executing notebooks on Colab, uncomment the cell saying:

    ::

        !pip install git+https://github.com/tee-lab/PyDaddy.git

    and execute it. This sets up PyDaddy on your Colab environment.


You can also download these notebooks from the `GitHub repo. <https://github.com/tee-lab/PyDaddy/tree/master/notebooks>`_. To download the notebooks, click on the |github| buttons to open the notebook. Right-click on the 'Raw' button to and click 'Save Linked File...' to save the file to your computer.

Getting started with PyDaddy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-getting-started| |github-getting-started|

This notebook introduces the basic functionalities of PyDaddy, using a 1-dimensional dataset. The notebook explores how to visually inspect drift and diffusion functions, how to fit analytical expressions to them, and how to use the various diagnostic tools provided.

Getting started with vector data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-vector| |github-vector|

PyDaddy also works with 2-dimensional vector data. This notebook demonstrates PyDaddy operation with a vector dataset.

Recovering SDEs from synthetic time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-synthetic| |github-synthetic|

This notebook generates a simulated time series from a user-specified SDE, and uses PyDaddy to recover the drift and diffusion functions from the simulated time series. Use this notebook to play around with the PyDaddy fitting procedure and gain insights.

Advanced function fitting
^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-advanced-fitting| |github-advanced-fitting|

PyDaddy can discover analytical expressions for the drift and diffusion functions. This notebook describes how to customize the fitting procedure to obtain best results.

Exporting data
^^^^^^^^^^^^^^
|colab-exporting| |github-exporting|

This notebook demonstrates how to export the recovered drift and diffusion data as CSV files or Pandas data-frames.

Fitting non-polynomial functions (Experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|colab-nonpoly-fitting| |github-nonpoly-fitting|

By default, PyDaddy fits polynomials for the drift and diffusion functions. However, this behaviour can be customized by providing a custom library of functions for the sparse regression procedure, this notebook demonstrates how to do this. (This is an experimental feature and not all functionality will work with non-polynomial functions).
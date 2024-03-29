
# PyDaddy

A Python package to discover stochastic differential equations from time series data.

[![Documentation Status](https://readthedocs.org/projects/pydaddy/badge/?version=latest)](https://pydaddy.readthedocs.io/en/latest/?badge=latest) [![](https://img.shields.io/github/license/tee-lab/PyDaddy) ](https://github.com/tee-lab/PyDaddy/blob/master/LICENSE.txt) [![](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org/abs/2205.02645)


PyDaddy is a comprehensive and easy to use python package to discover data-derived stochastic differential equations from time series data. PyDaddy takes the time series of state variable $x$, scalar or 2-dimensional vector, as input and discovers an SDE of the form:

$$ \frac{dx}{dt} = f(x) + g(x) \cdot \eta(t) $$

where $\eta(t)$ is Gaussian white noise. The function $f$ is called the _drift_, and governs the deterministic part of the dynamics. $g^2$ is called the _diffusion_ and governs the stochastic part of the dynamics.

| ![](https://github.com/tee-lab/PyDaddy/blob/master/resources/PyDaddyExample.jpg?raw=true) |
| :---: |
| An example summary plot generated by PyDaddy, for a vector time series dataset. |

PyDaddy also provides a range of functionality such as equation-learning for the drift and diffusion functions using sparse regresssion, a suite of diagnostic functions, etc.. For more details on how to use the package, check out the [example notebooks](./notebooks) and [documentation](https://pydaddy.readthedocs.io/).

| ![](https://github.com/tee-lab/PyDaddy/blob/master/resources/PyDaddySchematic.jpg?raw=true) |
| :---: |
| Schematic illustration of PyDaddy functionality. |

## Getting started

PyDaddy can be executed online on [Google Colab](https://colab.research.google.com/), without having to install it on your local machine. To run PyDaddy on Colab, open a notebook on Colab. Paste the following code on a notebook cell and run it:

    %pip install git+https://github.com/tee-lab/PyDaddy.git

This sets up PyDaddy in the notebook environment.

There are several example notebooks provided, which can be used to familiarize yourself with various features and functionalities of PyDaddy. These can be executed on Colab.

- [Getting started](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/1_getting_started.ipynb): Introduction to the basic functionalities of PyDaddy, using a 1-dimensional dataset.
- [Getting started with vector data](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/2_getting_started_vector.ipynb): Introduction to the basic functionalities of PyDaddy on 2-dimensional datasets.
- [Advanced function fitting](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/3_advanced_function_fitting.ipynb): PyDaddy can discover analytical expressions for the drift and diffusion functions. This notebook describes how to customize the fitting procedure to obtain best results.
- [Recovering SDEs from synthetic time series](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/4_sdes_from_simulated_timeseries.ipynb): This notebook generates a simulated time series from a user-specified SDE, and uses PyDaddy to recover the drift and diffusion functions from the simulated time series.
- [Exporting data](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/5_exporting_data.ipynb): Demonstrates how to export the recovered drift and diffusion data as CSV files or Pandas data-frames.
- [Fitting non-polynomial functions](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/6_non_poly_function_fitting.ipynb): PyDaddy fits polynomial functions to drift and diffusion by default. This behaviour can be customized, this notebook illustrates how to do this.

There are also two notebooks that use PyDaddy to discover SDEs from real-world datasets.
- [Example analysis - fish schooling](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/7_example_fish_school.ipynb): An example analysis of a fish schooling dataset (Jhawar et. al., Nature Physics, 2020) using PyDaddy. 
- [Example analysis - cell migration](https://colab.research.google.com/github/tee-lab/PyDaddy/blob/colab/notebooks/8_example_cell_migration.ipynb): An example analysis of a confined cell migration dataset (Brückner et. al., Nature Physics, 2019) using PyDaddy.

## Installation
PyDaddy is available both on PyPI and Anaconda Cloud, and can be installed on any system with a Python 3 environment. If you don't have Python 3 installed on your system, we recommend using [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). See the PyDaddy [package documentation](https://pydaddy.readthedocs.io/) for detailed installation instructions.

### Using pip
![PyPI](https://img.shields.io/pypi/v/pydaddy?color=blue) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/pydaddy) ![PyPI - Status](https://img.shields.io/pypi/status/pydaddy)

To install the latest stable release version of PyDaddy, use:

	pip install pydaddy

To install the latest development version of PyDaddy, use:

    pip install git+https://github.com/tee-lab/PyDaddy.git

### Using anaconda
![](https://anaconda.org/tee-lab/pydaddy/badges/version.svg) ![](https://anaconda.org/tee-lab/pydaddy/badges/latest_release_date.svg) ![](https://anaconda.org/tee-lab/pydaddy/badges/platforms.svg)

To install using `conda`, [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) need to be installed first. Once this is done, use the following command.

    conda install -c tee-lab pydaddy
    
<!---
   *Click [here](https://github.com/tee-lab/PyDaddy/archive/master.zip) to download source repository zip file.*
--->

## Documentation
For more information about PyDaddy, check out the [package documentation](https://pydaddy.readthedocs.io/).

## Citation
If you are using this package in your research, please cite the repository and the associated [paper](https://arxiv.org/abs/2205.02645) as follows:

Nabeel, A., Karichannavar, A., Palathingal, S., Jhawar, J., Danny Raj, M., & Guttal, V. (2022). PyDaddy: A Python Package for Discovering SDEs from Time Series Data (Version 1.1.0) [Computer software]. https://github.com/tee-lab/PyDaddy

Nabeel, A., Karichannavar, A., Palathingal, S., Jhawar, J., Danny Raj, M., & Guttal, V. (2022). PyDaddy: A Python package for discovering stochastic dynamical equations from timeseries data. arXiv preprint [arXiv:2205.02645](https://arxiv.org/abs/2205.02645).

## Licence
PyDaddy is distributed under the [**GNU General Public License v3.0**](./LICENSE.txt).


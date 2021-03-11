|Characterizing Noise|

pyddsde
=======

A package to derive an SDE form the data.

|Documentation Status| |image1| |Binder|
|enter image description here| |image2| |image3| |Conda|

pyddsde is a python package implementing a data driven SDE method.
pyddsde provides an interface which takes time series data as input,
runs the analysis and returns an output object through which data and
analysed results can be systematically visualized and saved.

How it works
============

Suppose m(t) is a SDE time series data.

   |image4|

The the package calculates the deterministic (drift) and stochastic
(diffusion) component of dm/dt.

======== ========
|image5| |image6|
======== ========
|image7| |image8|
======== ========

This data driven SDE method is based on the assumption that the noise in
the time series is uncorrelated and Gaussian in nature, with zero mean
and unit variance.

|image9|

pyddsde extracts the noise form the data and checks if it holds true to
its assumptions.

========= =========
|image10| |image11|
========= =========
|image12| |image13|
========= =========

Features
========

-  Simple one line execution of analysis.
-  Produces intuitive visuals of data and obtained result.
-  Supports time series data with both scalar and vector order
   parameters.
-  Generates interactive, drift and diffusion sliders for user defined
   timescale range or list.
-  Diagnostics help understand change in order of drift and diffusion
   with varying timescale.

Requirements
============

Python:

-  python version >=3.5 , <=3.8

Packages dependencies:

-  statsmodels, version 0.11.1
-  matplotlib, version 3.2.2
-  scipy, version 1.5.2
-  numpy, version 1.19.1
-  seaborn, version 0.10.1
-  tqdm, version 4.48.2
-  plotly, version 4.14.3

Installation
============

pyddsde is available both on pypi and anaconda cloud, which offers two
recommended methods of installations, using pip python package manager
or anaconda package manager.

.. note::
    To run the example notebooks on your system after installing
    the package, please download the notebook files to your PC from the
    `github <https://github.com/tee-lab/pyddsde>`__ repo.

    Alternately, its recommended to simply clone or download git the
    repository.

Installing via conda
~~~~~~~~~~~~~~~~~~~~

|image14| |enter image description here| |image15| |Conda| |image16|

`Conda <https://conda.io/docs/>`__ is a package manager that handles
sourcing of all dependencies in a relatively straight-forward,
cross-platform manner. pyddsde is available on the ``tee-lab`` channel.

.. important::
   Make sure you have anaconda or miniconda installed and have activated
   the conda default (base) environment using ``conda activate base`` ,
   before proceeding.

Install pyddsde package with all its dependencies using conda simply
requires executing the command

::

   conda install -c tee-lab pyddsde

To install pyddsde in a clean virtual environment use

::

    conda create --name MY_ENV_NAME -c tee-lab pyddsde

Replace ``MY_ENV_NAME`` with your desired name for environment.

You can now access pyddsde package by activating the newly created
environment, ``conda activate MY_ENV_NAME``

Installing via pip
~~~~~~~~~~~~~~~~~~

To install using pip package manager, run:

::

   python -m pip install pyddsde

Manual installation
~~~~~~~~~~~~~~~~~~~

.. sidebar:: Installing Latest unreleased version

    Without git, you will need to download the zip-file of the code, extract
    it and follow the above instructions.

    *Click* `here <https://github.com/tee-lab/pyddsde/archive/master.zip>`__ *to
    download source code zip file.*

This method is not recommended unless you experience problems with conda
or pip. To install using setuptools, download the source code manually
and run ``python setup.py install`` from the terminal.

This will install the package in your current environment (if you are
working in any environment).

If you have git installed, you can clone the repo and install using the
following commands.

::

   $ git clone https://github.com/tee-lab/pyddsde.git
   $ cd pyddsde
   $ python setup.py install

.. important::

    Without git, you will need to download the zip-file of the code, extract
    it and follow the above instructions.

    *Click* `here <https://github.com/tee-lab/pyddsde/archive/master.zip>`__ *to
    download source code zip file.*

Usage
=====

The time series data is given as input to the ``Characterize`` method
along with all other optional parameters.

.. toggle-header::
    :header: Show ``pyddsde.Characterize`` documentation

    .. admonition:: Characterize

      .. autoclass:: pyddsde.Characterize
         :members:
         :noindex:


Key parameters
~~~~~~~~~~~~~~

   data: list
       time series data to be analysed, 
       data = [x] for scalar data and data = [x1, x2] for vector
       where x, x1 and x2 are of numpy.array types.
   t: array or float
       float if its time increment between observation
       numpy.array if time stamp of time series

See doc strings or code documentation for more information.


Example using sample data set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`Data set description` for more information about the datasets.

::

   import pyddsde
   #load data
   data, t = pyddsde.load_sample_dataset('model-data-vector-ternary')
   # Analyse
   ddsde = pyddsde.Characterize(data,t)

   # Show drift slider plot
   ddsde.drift()
   # Show diffuision slider plot
   ddsde.diffusion()
   # Show timeseries plot
   ddsde.timeseries()
   # Show histograms
   ddsde.histograms()
   # Show all inputed, calculated and assumed parameters of the analysis
   ddsde.parameters()
   # Export data to disk
   ddsde.export_data()

``Characterize`` returns an output object in which all analysed results
are stored. Results can be visualised or stored by calling appropriate
functions:

.. toggle-header::
    :header: Show ``pyddsde.output.output`` documentation

    .. admonition:: pyddsde.output.output

      .. autoclass:: pyddsde.output
         :members:
         :noindex:

-  ``summary()``: show summary
-  ``drift()`` : drift slider plot
-  ``diffusion()`` : diffusion slider plot
-  ``timeseries()``: time series plot
-  ``histograms()`` : histogram plots
-  ``noise_characterstics()``: noise characteristics plots
-  ``visualise(timescale)``: drift and diffusion plots for a timescale
-  ``diagnostics()``: diagnostics plots
-  ``data(timescale)``: get drift and diffusion data for a timescale
-  ``export_data()``: Save data as csv files and mat files
-  ``plot_data(data)``: plot data on a 3d axis

For more examples see
`this <https://nbviewer.jupyter.org/github/tee-lab/pyddsde/blob/master/notebooks/Examples.ipynb>`__
notebook.

|badge|

Motivation
==========

|image17|

This project is motivated by the study of group behaviour
dynamics of animals, especially schooling fish. The sample data
distributed along with this package is from experiments conducted by
`TEElab, IISc <https://teelabiisc.wordpress.com/>`__.

Data set description
====================

pyddsde has six data set included along with the package which can be
loaded using ``load_sample_dataset(dataset_name)`` function

.. toggle-header::
    :header: Show ``pyddsde.load_sample_dataset`` documentation

    .. admonition:: pyddsde.load_sample_dataset

      .. autoclass:: pyddsde.load_sample_dataset
         :members:
         :noindex:

Experiment data (from experimentation or from observations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  fish-data-ectropus: 
    A data from experiment conducted with a group of
    30 fish, in which the group polarity in x and y directions are recorded
    every 0.12 seconds.

.. admonition:: Source

  - The fish data is a part of the work done in the *Noise-Induced Schooling of Fish*

Simulation data
~~~~~~~~~~~~~~~

A synthetic data set obtained from the simulation of fish interactions.

  model-data-vector-pairwise : 
    Pairwise interaction of fish simulated
    in two dimension.

  model-data-vector-ternary:
    Ternary interaction of fish simulated in
    two dimension.

  model-data-scalar-pairwise:
    Pairwise interaction of fish simulated in single dimension.

  model-data-scalar-ternary: 
    Ternary interaction of fish simulated in
    single dimension.

.. admonition:: Source

  - The simulation method is inspired from the work done in *Noise-induced Effects in Collective Dynamics and Inferring Local Interactions from Data*

Glossary
========

 .. glossary::

   drift
      First jump moment

   diffusion
      second jump moment


Acknowledgement
===============

Licence
=======

Distributed under **GNU General Public License v3.0**. See
``Licence.txt`` for more information.

Reference
=========

[1] Noise-induced Effects in Collective Dynamics and Inferring Local
Interactions from Data `Preprint <https://arxiv.org/abs/1911.09376>`__
`Github <https://github.com/tee-lab/Characterizing_noise>`__

[2] Noise-Induced Schooling of Fish
`Preprint <https://arxiv.org/abs/1903.12132>`__
`Github <https://github.com/tee-lab/schooling_fish>`__

.. |Characterizing Noise| image:: https://raw.githubusercontent.com/ashwinkk23/Characterizing_noise/master/.fig.jpg
.. |Documentation Status| image:: https://readthedocs.org/projects/pyddsde/badge/?version=latest
   :target: https://pyddsde.readthedocs.io/en/latest/?badge=latest
.. |image1| image:: https://img.shields.io/github/license/tee-lab/pyFish
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/tee-lab/pyFish.git/master?filepath=notebooks
.. |Conda| image:: https://img.shields.io/conda/pn/ashwinkk23/pyddsde
.. |enter image description here| image:: https://anaconda.org/ashwinkk23/pyddsde/badges/version.svg
.. |image2| image:: https://anaconda.org/ashwinkk23/pyddsde/badges/latest_release_date.svg
.. |image3| image:: https://anaconda.org/ashwinkk23/pyddsde/badges/installer/conda.svg
.. |image4| image:: https://latex.codecogs.com/gif.latex?\frac{dm}{dt}=f(m)+g(m)\eta(t)
.. |image5| image:: https://latex.codecogs.com/gif.latex?f(m)
.. |image6| image:: https://latex.codecogs.com/gif.latex?g(m)\eta(t)
.. |image7| image:: https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/deterministic.png?raw=true
.. |image8| image:: https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/stochastic.png?raw=True
.. |image9| image:: https://latex.codecogs.com/gif.latex?<\eta(t)>=0;<\eta(t)\eta(t')>=\delta(t-t')
.. |image10| image:: https://latex.codecogs.com/gif.latex?<\eta(t)>=0
.. |image11| image:: https://latex.codecogs.com/gif.latex?<\eta(t)\eta(t')>=\delta(t-t')
.. |image12| image:: https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/Test_of_hypothesis.png?raw=true
.. |image13| image:: https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/Noise_ACF.png?raw=true
.. |image14| image:: https://anaconda.org/ashwinkk23/pyddsde/badges/installer/conda.svg
.. |image15| image:: https://anaconda.org/ashwinkk23/pyddsde/badges/latest_release_date.svg
.. |image16| image:: https://anaconda.org/ashwinkk23/pyddsde/badges/license.svg
.. |badge| image:: https://img.shields.io/badge/run%20notebook-binder-E66581.svg?logo=data:image/png;base64,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
   :target: https://mybinder.org/v2/gh/tee-lab/pyFish/master?filepath=notebooks/Examples.ipynb
.. |image17| image:: https://teelabiisc.files.wordpress.com/2019/03/cropped-fish-7.jpg

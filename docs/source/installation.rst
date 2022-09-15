Installation Guide
==================

Prerequisites
-------------
You need to have a working Python 3 environment on your system. The recommended way to get a Python 3 installation is using Miniconda (minimal installation, very small download) or Anaconda (comes with many packages pre-installed, so a much larger ~3GB download: but a bit more beginner-friendly to install). See the respective pages for more information.

https://www.anaconda.com

https://docs.conda.io/en/latest/miniconda.html

https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda

.. note::
    If you are comfortable working with Conda environments, we recommend creating a new environment for PyDaddy, but this is not strictly necessary. For more information about environments, see: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

If you are installing the development version (see below), also ensure that you have Git installed on your system. You can do this using the following command:

::

    git --version

(If Git is already installed, this should show the version number; otherwise, it will show an error message saying git is not found.)

If Git is not installed, download and install it from https://git-scm.com

.. note::

    On macOS/Linux, the following commands needs to be entered in the Terminal. On a Windows machine with Anaconda/Miniconda, these need to be entered into the 'Anaconda Terminal' app.

Installation using :code:`pip`
------------------------------

Once Python 3 is set up, the latest release version can be installed using the following command:

::

    pip install pydaddy


To install the latest development version of PyDaddy directly from the GitHub repo, use:

::

    pip install git+https://github.com/tee-lab/PyDaddy.git

Installation using :code:`conda`
--------------------------------

The release version of PyDaddy can also be installed using conda as follows:

::

    conda install -c tee-lab pydaddy

Verifying installation
----------------------

To verify that everything is installed and working correctly, close and reopen the Terminal (Anaconda Terminal if you are using Windows) and try the following. First open an IPython terminal as follows:

::

    ipython

Within IPython, type the following:

::

    import pydaddy

If the installation was sucessful, the import command should work and throw no errors. Now type

::

    pydaddy.__version__

This should print the current version number of PyDaddy.

Congratulations, you have now successfully installed PyDaddy!



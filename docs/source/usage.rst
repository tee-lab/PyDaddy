Usage Guide
===========

There are multiple ways to use PyDaddy, and this page gives an overview of all of them.

PyDaddy on Google Colab
-----------------------

The easiest way to get started with PyDaddy is using Google Colaboratory (also called Colab).
Colab is a free Jupyter notebook environment, by Google, hosted entirely in the cloud. You can run PyDaddy on `Google Colab <https://colab.research.google.com>`_ notebooks, without having to install anything on your system. The :ref:`tutorial notebooks <tutorials>` have links to open them directly on Colab. (Note that you will need a Google login.)

To use PyDaddy from *any* Colab notebook, enter the following line to a cell and run it. This command sets up PyDaddy on the notebook's environment.

::

    %pip install git+https://github.com/tee-lab/PyDaddy.git

.. note::

    To upload your data files to the Colab notebook, click the 'Files' icon on the sidebar, and then click the 'Upload to session storage' icon.


.. warning::

    All files and data will be lost when you disconnect from the notebook. Make sure you download and save any relevant analysis results.

One-line operation
------------------

Once PyDaddy is installed on your machine, it can be invoked from the command-line using the single-command mode. This mode runs all relevant analysis on a specified data file, and generates a single HTML report with all the analysis results.

.. note::

    If you installed PyDaddy in a separate conda environment, activate that environment before continuing.

To use PyDaddy in this mode, use the following command:

::

    pydaddy <file-name> --column_format xyt

Replace :code:`<file-name>` with the name of the CSV file containing the data to be analyzed. The CSV file should contain the one or two data columns and one optional time-stamp column. The columns could be in any order; and the column order can be speficied using the :code:`--column_format` option as shown above. (For example, if the time-stamp column comes first followed by the x and y columns, the column format should be :code:`txy`.)
If time-stamp column is not present, the sampling interval can be provided using the :code:`-t` option.

For more details about other options and flags, use

::

    pydaddy --help

.. note::

    Ideally, the one-line functionality should be used only for a quick preliminary analysis. In particular, the results of the function fitting may not be optimal and may contain spurious terms. For best results, use PyDaddy within a notebook or script to fine-tune the estimation procedure (see the :ref:`advanced function fitting tutorial <tutorials>`).

Python Interface
----------------

.. note::

    If you installed PyDaddy in a separate conda environment, activate that environment before continuing.

For full control over the estimation procedure, you can use PyDaddy through Python scripts or notebooks. To use PyDaddy in a Jupyter notebook, start a Jupyter notebook server using the following command:

::

    jupyter notebook

Create a new notebook and import PyDaddy as follows.

::

    import pydaddy

You should be able to use all features of PyDaddy in the notebook. See the :ref:`tutorials <tutorials>` or :ref:`package documentation <package documentation>` for more details on available functionality.

pyddsde Installation Guide
==========================

Anaconda
--------

.. _this-guide-explains-creating-and-installing-pyfish-package:

This guide explains creating and installing pyddsde package.


If you don't have Anaconda installed, you can do so from `Anaconda Website <https://www.anaconda.com/products/individual>`_


.. _step-1--clone-the-git-repo:

Step 1 : Clone the git repo
'''''''''''''''''''''''''''

Open the terminal in your prefered directory and execute the below command to clone the repo

.. _git-clone-httpsgithubcomtee-labpyfishgit:

``git clone https://github.com/tee-lab/pyFish.git``


|enter image description here|

After cloning the terminal should look like this

|image1|


.. _step-2--change-the-directory-to-pyddsde:

Step 2 : Change the directory to ``pyddsde``
''''''''''''''''''''''''''''''''''''''''''''

``cd pyddsde``


|image2|

Typing ``ls`` should show the following content


|image3|

.. _step-3--create-python-environment:

Step 3 : Create python environment
''''''''''''''''''''''''''''''''''

.. _conda-env-create--f-environmentyml:

``conda env create -f environment.yml``

|image4|

Now, an environment named ``pyddsde`` should be created

|image5|


.. _step-4--activate-pyddsde-environment:

Step 4 : Activate pyddsde environment
'''''''''''''''''''''''''''''''''''''

``conda activate pyddsde``

.. _the-pyddsde-should-appear-in-the-terminal:

The (pyddsde) should appear in the terminal.

|image6|

.. _step-5--install-pyddsde:

Step 5 : Install pyddsde
''''''''''''''''''''''''

.. _python--m-pip-install-:

``python -m pip install .``

|image7|

If you see a similar output at the end then the package is successfully installed


|image8|


You can run the notebook files using jupyter notebook (or jupyter lab)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

``jupyter notebook``


|image9|

.. _this-should-open-the-notebook-application-in-the-browser-click-on-notebooks-folder-and-open-the-ipynb-notebook-file:

This should open the notebook application in the browser, click on ``notebooks`` folder and open the .ipynb notebook file.

.. _after-opening-the-file-click-on-the-cell-and-press-shiftenter-to-execute-that-cell-and-move-to-the-next:

After opening the file, click on the cell and press ``Shift+Enter`` to execute that cell and move to the next.


pip
---

.. |enter image description here| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/git_clone.png?raw=true
.. |image1| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/git_clone2.png?raw=true
.. |image2| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/cd_pyfish.png?raw=true
.. |image3| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/ls1.png?raw=true
.. |image4| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/env_create1.png?raw=true
.. |image5| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/env_create2.png?raw=true
.. |image6| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/activate_pyFish.png?raw=true
.. |image7| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/pip_install1.png?raw=true
.. |image8| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/pip_install2.png?raw=true
.. |image9| image:: https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/jupyter_nb.png?raw=true

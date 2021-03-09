# `pyddsde` 

---

# Installation Guide
### This guide explains creating and installing pyFish package.
### If you don't have Anaconda installed, you can do so from [`Anaconda Website`](https://www.anaconda.com/products/individual)

---

# Step 1 : Clone the git repo

### Open the terminal in your prefered directory and execute the below command to clone the repo
### `git clone https://github.com/tee-lab/pyFish.git`

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/git_clone.png?raw=true)

### After cloning the terminal should look like this

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/git_clone2.png?raw=true)

---

# Step 2 : Change the directory to `pyddsde`

### `cd pyddsde`

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/cd_pyfish.png?raw=true)

### Typing `ls` should show the following content

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/ls1.png?raw=true)

---

# Step 3 : Create python environment

## `conda env create -f environment.yml`

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/env_create1.png?raw=true)

### Now, an environment named `pyddsde` should be created

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/env_create2.png?raw=true)
---

# Step 4 : Activate pyddsde environment

## `conda activate pyddsde`

### The (pyddsde) should appear in the terminal.

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/activate_pyFish.png?raw=true)

---

# Step 5 : Install pyddsde

## `python -m pip install .`

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/pip_install1.png?raw=true)

### If you see a similar output at the end then the package is successfully installed

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/pip_install2.png?raw=true)

---

# You can run the notebook files using jupyter notebook (or jupyter lab)

### `jupyter notebook`

![enter image description here](https://github.com/tee-lab/pyFish/blob/master/notebooks/imgs/jupyter_nb.png?raw=true)

### This should open the notebook application in the browser, click on `notebooks` folder and open the .ipynb notebook file. 
### After opening the file, click on the cell and press `Shift+Enter` to execute that cell and move to the next.
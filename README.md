
![Characterizing Noise](https://raw.githubusercontent.com/ashwinkk23/Characterizing_noise/master/.fig.jpg)
# pyddsde

A package to derive an SDE form the data.

[![Documentation Status](https://readthedocs.org/projects/pyddsde/badge/?version=latest)](https://pyddsde.readthedocs.io/en/latest/?badge=latest)  ![](https://img.shields.io/github/license/tee-lab/pyFish)


pyddsde is a python package implementing a data driven SDE method. pyddsde provides an interface which takes time series data as input, runs the analysis and returns an output object through which data and analysed results can be systematically visualized and saved.

Suppose m(t) is a SDE time series data.

>![](https://latex.codecogs.com/gif.latex?\frac{dm}{dt}=f(m)+g(m)\eta(t))

The the package calculates the deterministic (drift) and stochastic (diffusion) component of dm/dt.

![](https://latex.codecogs.com/gif.latex?f(m))          |  ![](https://latex.codecogs.com/gif.latex?g(m)\eta(t))
:-------------------------:|:-------------------------:
![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/deterministic.png?raw=true)  |  ![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/stochastic.png?raw=True)


This data driven SDE method is  based on the assumption that the noise in the time series is uncorrelated and Gaussian in nature, with zero mean and unit variance.
 
  ![]( https://latex.codecogs.com/gif.latex?<\eta(t)>=0;<\eta(t)\eta(t')>=\delta(t-t'))
 
 pyddsde extracts the noise form the data and checks if it holds true to its assumptions.
 > 
![](https://latex.codecogs.com/gif.latex?<\eta(t)>=0)|  ![](https://latex.codecogs.com/gif.latex?<\eta(t)\eta(t')>=\delta(t-t'))
:-------------------------:|:-------------------------:
![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/Test_of_hypothesis.png?raw=true)  |  ![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/Noise_ACF.png?raw=true)


# Installation
To install and run the provided notebook files, clone the repo and install so you have a copy of all the notebook files provided.

    $ git clone https://github.com/tee-lab/pyddsde.git
    $ cd pyFish
    $ python -m pip install .
    
To just install the package use:

    $ python -m pip install git+https://github.com/tee-lab/pyddsde.git
  
# Usage
The time series data is given as input to the `Characterize` method along with all other optional parameters.
#### Key parameters:

    data: list
		time series data to be analysed, 
		data = [x] for scalar data and data = [x1, x2] for vector
		where x, x1 and x2 are of numpy.array types.
	t : array or float
		float if its time increment between observation
		numpy.array if time stamp of time series
See doc strings or documentation for more information.

#### Example using sample data set

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

`Characterize`  returns an output object in which all analysed results are stored.
Results can be visualised or stored by calling appropriate functions:
`summary()`: show summary
`drift()` : drift slider plot
`diffusion()` : diffusion slider plot
`timeseries()`:  time series plot
`histograms()` : histogram plots
`noise_characterstics()`: noise characteristics plots
`visualise(timescale)`: drift and diffusion plots for a timescale
`diagnostics()`: diagnostics plots
`data(timescale)`: get drift and diffusion data for a timescale
`export_data()`: Save data as csv files and mat files
`plot_data(data)`: plot data on a 3d axis

For more examples see [this](https://github.com/tee-lab/pyFish/blob/master/notebooks/Examples.ipynb) notebook.

*For more info refer the docs page*
# Motivation
![](https://teelabiisc.files.wordpress.com/2019/03/cropped-fish-7.jpg)
This project is motivated by the study of group behaviour dynamics of animals, especially schooling fish.
The sample data distributed along with this package is from experiments conducted by [TEElab, IISc](https://teelabiisc.wordpress.com/).

# Data set description
pyddsde has six data set included along with the package which can be loaded using `load_sample_dataset(dataset_name)` function

### Experiment data (from experimentation or from observations)
`fish-data-ectropus`:
A data from experiment conducted with a group of 30 fish, in which the group polarity in x and y directions are recorded every 0.12 seconds. 
> This data is a part of the work done in the *Noise-Induced Schooling of Fish*
### Simulation data
A synthetic data set obtained from the simulation of fish interactions.

`model-data-vector-pairwise` : 
Pairwise interaction of fish simulated in two dimension.

`model-data-vector-ternary`: 
Ternary interaction of fish simulated in two dimension.

`model-data-scalar-pairwise`:  
Pairwise interaction of fish simulated in single dimension.

`model-data-scalar-ternary`: 
 Ternary interaction of fish simulated in single dimension.

> The simulation method is inspired from the work done in *Noise-induced Effects in Collective Dynamics and Inferring Local Interactions from Data*


# Licence
Distributed under **GNU General Public License v3.0**. See `Licence.txt` for more information.


# Reference
[1] Noise-induced Effects in Collective Dynamics and Inferring Local Interactions from Data
   [Preprint](https://arxiv.org/abs/1911.09376) [Github](https://github.com/tee-lab/Characterizing_noise)
 
[2] Noise-Induced Schooling of Fish 
	 [Preprint](https://arxiv.org/abs/1903.12132) [Github](https://github.com/tee-lab/schooling_fish) 

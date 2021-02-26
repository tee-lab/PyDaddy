
![Characterizing Noise](https://raw.githubusercontent.com/ashwinkk23/Characterizing_noise/master/.fig.jpg)
# Introduction
![](https://img.shields.io/badge/numpy-1.19.1-green)    ![](https://img.shields.io/badge/scipy-1.5.2-blue) ![](https://img.shields.io/badge/statsmodels-0.11.1-yellow) ![](https://img.shields.io/badge/matplotlib-3.2.2-red) ![](https://img.shields.io/badge/tqdm-4.48.2-lightgrey)  ![](https://img.shields.io/badge/seaborm-0.10.1-orange)

Package to analyse the characteristics of stochastic time series data.

# How does it work
Suppose m(t) is a stochastic time series data, then the package calculates the deterministic and stochastic component of dm/dt.

>![](https://latex.codecogs.com/gif.latex?\frac{dm}{dt}=f(m)+g(m)\eta(t))

![](https://latex.codecogs.com/gif.latex?f(m))          |  ![](https://latex.codecogs.com/gif.latex?g(m)\eta(t))
:-------------------------:|:-------------------------:
![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/deterministic.png?raw=true)  |  ![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/stochastic.png?raw=True)


The method of extracting the deterministic and stochastic and deterministic functions is based on the assumption that the noise is uncorrelated and Gaussian in nature.
 > ![]( https://latex.codecogs.com/gif.latex?<\eta(t)>=0;<\eta(t)\eta(t')>=\delta(t-t'))
 > 
![](https://latex.codecogs.com/gif.latex?\<eta(t)>=0)          |  ![](https://latex.codecogs.com/gif.latex?<\eta(t)\eta(t')>=\delta(t-t'))
:-------------------------:|:-------------------------:
![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/Test_of_hypothesis.png?raw=true)  |  ![](https://github.com/ashwinkk23/pyFish/blob/master/notebooks/imgs/Noise_ACF.png?raw=true)


The package extracts the noise from the data and checks to make sure the noise is Gaussian and uncorrelated in nature. 


# Motivation
![](https://teelabiisc.files.wordpress.com/2019/03/cropped-fish-7.jpg)
This project is motivated by the study of group behaviour dynamics of animals, especially schooling fish.
The sample data distributed along with this package is gathered from experiments conducted by [TEElab, IISc](https://teelabiisc.wordpress.com/).

# Installation

    $ git clone https://github.com/tee-lab/pyFish.git
    $ cd pyFish
    $ python setup.py install
or

    $ python -m pip install git+https://github.com/tee-lab/pyFish.git

For developers:
Clone the repo:

    $ git clone https://github.com/ashwinkk23/pyFish.git
    $ cd pyFish
    $ python setup.py build
    $ python setup.py develop

# Usage

    pyFish.Characterize(...)
    Input params:
		data : list
			time series data to be analysed, data = [x] for scalar data and data = [x1, x2] for vector
			where x, x1 and x2 are of numpy.array object type
		t : numpy.array
			time stamp of time series
		t_int : float
			time increment between consecutive observations of the time series
		dt = 'auto' : 'auto' or int
			time scale to run the analysis on (for determinsitic part);
			algorithm estimates dt if 'auto' is passed, else takes the user input
		delta_t = 1 : int
			time scale to run the analysis on (for stochastic part)
		inc = 0.01 : float
			increment in order parameter for scalar data
		inc_x = 0.1 : float
			increment in order parameter for vector data x1
		inc_y = 0.1 : float
			increment in order parameter for vector data x2
		drift_order = None : int
			order of polynomial to be fit for calculated drift (deterministic part);
			if None, algorithim estimates the optimium drift_order
		diff_order = None : int
			order of polynomial to be fit for calculated diff (stochastic part);
			if None, algorithim estimates the optimium diff_order
		max_order = 10 : int
			maxmium drift_order and diff_order to consider
		fft = True : bool
			if true use fft method to calculate autocorrelation else, use standard method
		t_lag = 1000 : int
			maxmium lag to use to calculate acf
		**kwargs 
			All the parameters for pyFish.preporcessing and pyFish.noise_analysis


# Usage Example
Example loading sample data and storing all data and plots.

    # import modules
    >>> import pyFish
    >>> import pyFish.tests
    # load data
    >>> data = pyFish.tests.load_sample_data('data/ternary/N30.csv')
    >>> X, t = data.T
    # Analyse
    >>> analysed_output = pyFish.Characterize([X],t)
    Gaussian check for underlying noise: 100%|█| 10000/10000 [00:00<00:00, 14393.3]
    # Save analysed data, plots, parameters to disk.
    >>> analysed_output.save_all_data(savepath='ternary_n30')
    Results saved in: ternary_n30/1600538555
   
The results will be saved in ``current/working/directory/ternary_n30/`` folder

   *For more information and examples, please refer to the [Wiki](https://github.com/ashwinkk23/pyFish/wiki) page*

# Licence
Distributed under **GNU General Public License v3.0**. See `Licence.txt` for more information.


# Reference
* Noise-induced Effects in Collective Dynamics and Inferring Local Interactions from Data
  - [![](https://img.shields.io/badge/Preprint-arxiv-red)](https://arxiv.org/abs/1911.09376) [![](https://img.shields.io/badge/Characterizing_Noise-github-blue)](https://github.com/tee-lab/Characterizing_noise)
* Noise-Induced Schooling of Fish 
	- [![](https://img.shields.io/badge/Preprint-arxiv-red)](https://arxiv.org/abs/1903.12132) [![](https://img.shields.io/badge/schooling_fish-github-blue)](https://github.com/tee-lab/schooling_fish) 

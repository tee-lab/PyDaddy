import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.optimize
import scipy.stats
import statsmodels.api as sm 
import statsmodels.stats.diagnostic
from pyFish.preprocessing import preprocessing

class output(preprocessing):
	def __init__(self, X, t, drift, diff, avgdrift, avgdiff, op, drift_order, diff_order, variables,**kwargs):
		self.X = X
		self.t = t
		self.t_int= self.t[-1]/len(self.t)
		self.drift = drift
		self.diff = diff
		self.avgdrift = avgdrift
		self.avgdiff = avgdiff
		self.op = op
		self.drift_order = drift_order
		self.diff_order = diff_order
		self.variables = variables
		self.__dict__.update(kwargs)
		preprocessing.__init__(self)
	
	def data(self):
		return self.drift, self.diff, self.avgdrift, self.avgdiff, self.op

	def parameters(self):
		params = dict()
		for keys in self.variables.keys():
			if str(keys)[0] != '_':
				params[keys] = self.variables[keys]
		return params

	def visualize(self):
		#Time series
		fig1 = fig = plt.figure(dpi=150)
		l = int(len(self.X)/4)
		plt.plot(self.t[0:l],self.X[0:l])
		plt.title('Figure 1')
		#PDF
		fig2 = fig = plt.figure(dpi=150, figsize=(5,5))
		sns.distplot(self.X)
		plt.title('Figure 2')
		plt.xlim([-1,1])
		plt.ylabel('PDF')
		plt.xlabel('Order Parameter')
		#Drift
		fig3 = fig = plt.figure(dpi=150,figsize=(5,5))
		p_drift, _ = self.fit_poly(self.op, self.avgdrift, self.drift_order)
		plt.scatter(self.op, self.avgdrift, marker='.')
		plt.scatter(self.op, p_drift(self.op), marker='.')
		plt.title('Figure 3')
		plt.xlabel('Order Parameter')
		plt.ylabel("Deterministic")
		plt.xlim([-1,1])
		#Diffusion
		fig4 = fig = plt.figure(dpi=150,figsize=(5,5))
		p_diff, _ = self.fit_poly(self.op, self.avgdiff, self.diff_order)
		plt.scatter(self.op, self.avgdiff, marker='.')
		plt.scatter(self.op, p_diff(self.op), marker='.')
		plt.title('Figure 4')
		plt.xlim([-1,1])
		plt.xlabel("Order Parameter")
		plt.ylabel('Stochastic')
		plt.show()

	def diagnostic(self):
		#ACF
		fig1 = plt.figure(dpi=150)
		exp_fn = lambda t,a,b: a*np.exp((-1/b)*t)
		plt.plot(self.variables['_autocorr_x'], self.variables['_autocorr_y'])
		y = exp_fn(self.variables['_autocorr_x'], self.variables['_a'], self.variables['autocorrelation_time'])
		plt.plot(self.variables['_autocorr_x'], y)
		plt.legend(('ACF', 'exp_fit'))
		plt.title('Autocorrelation Function')
		plt.xlabel('Time Lag')
		plt.ylabel('ACF')
		#R2 vs order for drift
		fig2 = plt.figure(dpi=150)
		plt.plot(range(self.variables['max_order']), self.variables['_r2_drift'])
		plt.xlabel('order')
		plt.ylabel('R2')
		plt.title('R2 Drift vs order')
		#R2 vs order for diff
		fig3 = plt.figure(dpi=150)
		plt.plot(range(self.variables['max_order']), self.variables['_r2_diff'])
		plt.xlabel('order')
		plt.ylabel('R2')
		plt.title('R2 Diff vs order')
		plt.show()

	def histogram3d(self,x,bins = 10, normed = False, color = 'blue', alpha = 1, hold = False):
		"""
		Plotting a 3D histogram

		Parameters
		----------

		sample : array_like.		
		    The data to be histogrammed. It must be an (N,2) array or data 
		    that can be converted to such. The rows of the resulting array 
		    are the coordinates of points in a 2 dimensional polytope.

		bins : sequence or int, optional, default: 10.
		    The bin specification:
		    
		    * A sequence of arrays describing the bin edges along each dimension.
		    * The number of bins for each dimension (bins =[binx,biny])
		    * The number of bins for all dimensions (bins = bins).

		normed : bool, optional, default: False.
		    If False, returns the number of samples in each bin. 
		    If True, returns the bin density bin_count / sample_count / bin_volume.

		color: string, matplotlib color arg, default = 'blue'

		alpha: float, optional, default: 1.
		    0.0 transparent through 1.0 opaque

		hold: boolean, optional, default: False

		Returns   
		--------
		H : ndarray.
		    The bidimensional histogram of sample x.

		edges : list.
		    A list of 2 arrays describing the bin edges for each dimension.
		    
		See Also 
		--------
		histogram: 1-D histogram
		histogram2d: 2-D histogram
		histogramdd: N-D histogram

		Examples
		--------
		>>> r = np.random.randn(1000,2)
		>>> H, edges = np.histogram3d(r,bins=[10,15])
		"""

		if np.size(bins) == 1:
		    bins = [bins,bins]

		if(len(x) == 2):
		    x = x.T;
		    

		H, edges = np.histogramdd(x, bins, normed = normed)

		H = H.T
		X = np.array(list(np.linspace(min(edges[0]),max(edges[0]),bins[0]))*bins[1])   
		Y = np.sort(list(np.linspace(min(edges[1]),max(edges[1]),bins[1]))*bins[0])    

		dz = np.array([]);

		for i in range(bins[1]):
		    for j in range(bins[0]):
		        dz = np.append(dz, H[i][j])

		Z = np.zeros(bins[0]*bins[1])

		dx = X[1] - X[0]   
		dy = Y[bins[0]] - Y[0]

		if (not hold):
		    fig = plt.figure(dpi = 300)
		    ax = fig.add_subplot(111, projection='3d')
		    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
		else:
		    try:
		        ax = plt.gca();
		        ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
		    except:
		        plt.close(plt.get_fignums()[-1])
		        fig = plt.figure()
		        ax = fig.add_subplot(111, projection='3d')
		        ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
		        
		        
		plt.xlabel('X');
		plt.ylabel('Y');
		edges = [X,Y];
		H = dz.reshape(bins[0],bins[1]);

		return H, edges;
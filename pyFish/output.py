import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.optimize
import scipy.stats
import statsmodels.api as sm 
import statsmodels.stats.diagnostic
from pyFish.preprocessing import preprocessing

class output(preprocessing):
	def __init__(self, out, **kwargs):
		self.vector = out.vector

		if not self.vector:
			self.X = out._X
			self.t = out._t
			#self.t_int= self.t[-1]/len(self.t)
			self.drift = out._drift
			self.diff = out._diff
			self.avgdrift = out._avgdrift
			self.avgdiff = out._avgdiff
			self.op = out._op
			self.drift_order = out.drift_order
			self.diff_order = out.diff_order
		else:
			self.vel_x = out._vel_x
			self.vel_y = out._vel_y
			self.avgdriftX = out._avgdriftX
			self.avgdriftY = out._avgdriftY
			self.avgdiffX = out._avgdiffX
			self.avgdiffY = out._avgdiffY
			self.avgdiffXY = out._avgdiffXY
			self.op_x = out._op_x
			self.op_y = out._op_y

		self.out = out

		self.__dict__.update(kwargs)
		preprocessing.__init__(self)
	
	def data(self):
		if not self.vector:
			return self.drift, self.diff, self.avgdrift, self.avgdiff, self.op
		return self.avgdriftX, self.avgdriftY, self.avgdiffX, self.avgdiffY, self.avgdiffXY, self.op_x, self.op_y

	def parameters(self):
		params = dict()
		for keys in self.out.__dict__.keys():
			if str(keys)[0] != '_':
				params[keys] = self.out.__dict__[keys]
		return params

	def visualize(self):
		if not self.vector:
			#Time series
			fig1 = fig = plt.figure(dpi=150)
			l = int(len(self.X)/4)
			try:
				plt.plot(self.t[0:l],self.X[0:l])
			except:
				plt.plot(self.X[0:l])
			plt.title('Figure 1')
			#PDF
			fig2 = fig = plt.figure(dpi=150, figsize=(5,5))
			sns.distplot(self.X)
			plt.title('Figure 2')
			plt.xlim([min(self.X),max(self.X)])
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
			plt.xlim([min(self.X),max(self.X)])
			#Diffusion
			fig4 = fig = plt.figure(dpi=150,figsize=(5,5))
			p_diff, _ = self.fit_poly(self.op, self.avgdiff, self.diff_order)
			plt.scatter(self.op, self.avgdiff, marker='.')
			plt.scatter(self.op, p_diff(self.op), marker='.')
			plt.title('Figure 4')
			plt.xlim([min(self.X),max(self.X)])
			plt.xlabel("Order Parameter")
			plt.ylabel('Stochastic')
			plt.show()
		else:
			fig1 = plt.figure()
			ax = fig1.add_subplot(111, projection='3d')
			vel_x = self.interploate_missing(self.vel_x)
			vel_y = self.interploate_missing(self.vel_y)
			H, edges, X, Y, Z, dx, dy, dz = self.histogram3d(np.array([vel_x[~np.isnan(vel_x)], vel_y[~np.isnan(vel_y)]]))
			colors = plt.cm.jet(dz.flatten()/float(dz.max()))
			ax.bar3d(X,Y,Z,dx,dy,dz, alpha=1, color=colors)
			plt.xlim([1,-1])
			plt.title('Figure 1')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Frequency')

			fig2 = plt.figure()
			ax = fig2.add_subplot(projection="3d")
			x = np.matlib.repmat(self.op_x,len(self.op_x),1)
			x.ravel().sort()
			y = np.matlib.repmat(self.op_y, len(self.op_y),1)
			self.avgdiffY[self.avgdiffY==0] = np.nan
			z = self.avgdiffY
			ax.scatter3D(x, y, z.ravel(), c=z.ravel(), cmap='jet');
			plt.xlim([1,-1])
			plt.title('Figure 2')
			ax.set_xlabel('Mx')
			ax.set_ylabel('My')
			ax.set_zlabel('Stochastic Factor')

			fig3 = plt.figure()
			ax = fig3.add_subplot(projection="3d")
			x = np.matlib.repmat(self.op_x,len(self.op_x),1)
			x.ravel().sort()
			y = np.matlib.repmat(self.op_y, len(self.op_y),1)
			self.avgdiffX[self.avgdiffX==0] = np.nan
			z = self.avgdriftX
			ax.scatter3D(x, y, z.ravel(), c=z.ravel(), cmap='jet');
			plt.xlim([1,-1])
			plt.title('Figure 3')
			ax.set_xlabel('Mx')
			ax.set_ylabel('My')
			ax.set_zlabel('Stochastic Factor')

			fig4 = plt.figure()
			ax = fig4.add_subplot(projection="3d")
			x = np.matlib.repmat(self.op_x,len(self.op_x),1)
			x.ravel().sort()
			y = np.matlib.repmat(self.op_y, len(self.op_y),1)
			self.avgdriftY[self.avgdriftY==0] = np.nan
			z = self.avgdriftY
			ax.scatter3D(x, y, z.ravel(), c=z.ravel(), cmap='jet');
			plt.xlim([1,-1])
			plt.title('Figure 4')
			ax.set_xlabel('Mx')
			ax.set_ylabel('My')
			ax.set_zlabel('Deterministic Factor')

			fig5 = plt.figure()
			ax = fig5.add_subplot(projection="3d")
			x = np.matlib.repmat(self.op_x,len(self.op_x),1)
			x.ravel().sort()
			y = np.matlib.repmat(self.op_y, len(self.op_y),1)
			self.avgdriftX[self.avgdriftX==0] = np.nan
			z = self.avgdriftX
			ax.scatter3D(x, y, z.ravel(), c=z.ravel(), cmap='jet');
			plt.xlim([1,-1])
			plt.title('Figure 5')
			ax.set_xlabel('Mx')
			ax.set_ylabel('My')
			ax.set_zlabel('Deterministic Factor')
			plt.show()


	def diagnostic(self):
		t1 = "R2" if self.out.order_metric=="R2" else "R2_adj"
		#ACF
		fig1 = plt.figure(dpi=150)
		exp_fn = lambda t,a,b: a*np.exp((-1/b)*t)
		plt.plot(self.out._autocorr_x, self.out._autocorr_y)
		y = exp_fn(self.out._autocorr_x, self.out._a, self.out.autocorrelation_time)
		plt.plot(self.out._autocorr_x, y)
		plt.legend(('ACF', 'exp_fit'))
		plt.title('Autocorrelation Function')
		plt.xlabel('Time Lag')
		plt.ylabel('ACF')
		#R2 vs order for drift
		fig2 = plt.figure(dpi=150)
		plt.plot(range(self.out.max_order), self.out._r2_drift)
		plt.xlabel('order')
		plt.ylabel(t1)
		plt.title('{} Drift vs order'.format(t1))
		#R2 vs order for diff
		fig3 = plt.figure(dpi=150)
		plt.plot(range(self.out.max_order), self.out._r2_diff)
		plt.xlabel('order')
		plt.ylabel(t1)
		plt.title('{} Diff vs order'.format(t1))
		plt.show()

	def noise_characterstics(self):
		print("Noise is gaussian") if self.out.gaussian_noise else print("Noise is not Gaussian")
		fig1 = plt.figure(dpi=150)
		sns.distplot(self.out._noise)
		plt.title("Noise Distribution")
		fig2 = plt.figure(dpi=150)
		sns.distplot(self.out._kl_dist)
		start, stop = plt.gca().get_ylim()
		plt.plot(np.ones(len(self.out._X1))*self.out.l_lim, np.linspace(start,stop,len(self.out._X1)),'r', label='upper_cl')
		plt.plot(np.ones(len(self.out._X1))*self.out.h_lim, np.linspace(start,stop,len(self.out._X1)),'r', label="lower_cl")
		plt.plot(np.ones(len(self.out._X1))*self.out.k, np.linspace(start,stop,len(self.out._X1)),'g', label='Test Stat')
		plt.legend()
		plt.title("Test of hypothesis")
		fig3 = plt.figure(dpi=150)
		plt.plot(self.out._X1[1:], self.out._f)
		plt.plot(np.ones(len(self.out._X1[1:]))*self.out.l_lim, self.out._f, 'r', label='lower_cl')
		plt.plot(np.ones(len(self.out._X1[1:]))*self.out.h_lim, self.out._f, 'r', label='upper_cl')
		plt.plot(np.ones(len(self.out._X1[1:]))*self.out.k, self.out._f, 'g', label='Test Stat')
		plt.legend()
		plt.title("Cummulative Density Function")
		fig4 = plt.figure(dpi=150)
		plt.plot(self.out._noise_correlation[0], self.out._noise_correlation[1])
		plt.title("Noise ACF")
		plt.show()


	def histogram3d(self,x,bins = 10, normed = False, color = 'blue', alpha = 1, hold = False, plot_hist=False):
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

		if plot_hist:
			if (not hold):
			    fig = plt.figure(dpi = 300)
			    ax = fig.add_subplot(111, projection='3d')
			    colors = plt.cm.jet(dz.flatten()/float(dz.max()))
			    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = colors);
			else:
			    try:
			        ax = plt.gca();
			        colors = plt.cm.jet(dz.flatten()/float(dz.max()))
			        ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = colors);
			    except:
			        plt.close(plt.get_fignums()[-1])
			        fig = plt.figure()
			        ax = fig.add_subplot(111, projection='3d')
			        colors = plt.cm.jet(dz.flatten()/float(dz.max()))
			        ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = colors);
		        
		        
		plt.xlabel('X');
		plt.ylabel('Y');
		edges = [X,Y];
		H = dz.reshape(bins[0],bins[1]);

		#return H, edges;
		return H, edges, X, Y, Z, dx, dy, dz



class Error(Exception):
	"""Base class for exceptions in this module."""
	pass


class InputError(Error):
	"""Exception raised for errors in the input.

	Attributes:
	    expression -- input expression in which the error occurred
	    message -- explanation of the error
	"""

	def __init__(self, expression, message):
		self.expression = expression
		self.message = message
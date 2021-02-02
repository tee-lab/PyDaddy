import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyFish.metrics import metrics


class visualize(metrics):
    """
	plots
	"""
    def __init__(self, op_x, op_y, op, autocorrelation_time, **kwargs):
        self.__dict__.update(kwargs)
        self.op_x = op_x
        self.op_y = op_y
        self.op = op
        self.autocorrelation_time = int(autocorrelation_time)
        metrics.__init__(self)

    def _plot_summary(self,
                      data,
                      vector=True,
                      dpi=150,
                      kde=False,
                      start=0,
                      stop=1000):
        if not vector:
            M, drift, diff, drift_order, diff_order = data

            fig = plt.figure(dpi=dpi, figsize=(20, 20))

            plt.subplot(2, 2, 1)
            plt.plot(M[start:stop])
            plt.title('Timeseries', fontsize=16)
            plt.ylabel("|M|", fontsize=16)
            plt.xlabel('Time index', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.subplot(2, 2, 2)
            sns.distplot(M, kde=kde)
            plt.title("Dist M", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.subplot(2, 2, 3)
            p_drift, _ = self._fit_poly(self.op, drift, drift_order)
            plt.scatter(self.op, drift, marker='.')
            plt.scatter(self.op, p_drift(self.op), marker='.', alpha=0.4)
            plt.title("Drift", fontsize=16)
            plt.xlabel('Order Parameter', fontsize=16)
            plt.ylabel("$A_{1}$", fontsize=16)
            plt.xlim([min(M), max(M)])
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.subplot(2, 2, 4)
            p_diff, _ = self._fit_poly(self.op, diff, diff_order)
            plt.scatter(self.op, diff, marker='.')
            plt.scatter(self.op, p_diff(self.op), marker='.', alpha=0.4)
            plt.title("Diffusion", fontsize=16)
            plt.xlim([min(M), max(M)])
            plt.xlabel("Order Parameter", fontsize=16)
            plt.ylabel('$B_{11}$', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

        else:
            Mx, My, driftX, driftY, diffX, diffY = data
            M = np.sqrt(Mx**2 + My**2)

            fig = plt.figure(figsize=(30, 15))

            plt.subplot(2, 3, 1)
            plt.plot(M[start:stop])
            plt.title('TimeSeries', fontsize=16)
            plt.ylabel("|M|", fontsize=16)
            plt.xlabel('time index')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.subplot(2, 3, 4)
            sns.distplot(M, kde=kde)
            plt.title('Dist |M|', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            ax_driftX = plt.subplot(2, 3, 2, projection='3d')
            _, ax_driftX = self._plot_data(driftX,
                                           ax=ax_driftX,
                                           title=None,
                                           z_label='$A_{1}$')
            plt.title('Drift X', fontsize=16)

            ax_driftY = plt.subplot(2, 3, 3, projection='3d')
            _, ax_driftY = self._plot_data(driftY,
                                           ax=ax_driftY,
                                           title=None,
                                           z_label='$A_{2}$')
            plt.title('Drift Y', fontsize=16)

            ax_diffX = plt.subplot(2, 3, 5, projection='3d')
            _, ax_diffX = self._plot_data(diffX,
                                          ax=ax_diffX,
                                          title=None,
                                          z_label='$B_{11}$')
            plt.title('Diff X', fontsize=16)

            ax_diffY = plt.subplot(2, 3, 6, projection='3d')
            _, ax_diffY = self._plot_data(diffY,
                                          ax=ax_diffY,
                                          title=None,
                                          z_label='$B_{22}$')
            plt.title('Diff Y', fontsize=16)
        return fig

    def _plot_timeseries(self,
                         timeseries,
                         vector,
                         start=0,
                         stop=1000,
                         dpi=150):
        if vector:
            Mx, My = timeseries
            fig = plt.figure(dpi=dpi)
            fig.suptitle('Timeseries')
            plt.subplot(2, 1, 1)
            plt.plot(Mx[start:stop])
            plt.ylabel("$M_{x}$")
            plt.subplot(2, 1, 2)
            plt.plot(My[start:stop])
            plt.ylabel("$M_{y}$")
        else:
            Mx = timeseries[0]
            fig = plt.figure(dpi=dpi)
            plt.plot(Mx[start:stop])
            plt.title('Timeseries')
            plt.ylabel('M')

        return fig

    def _plot_histograms(self,
                         timeseries,
                         vector,
                         size=(20, 5),
                         dpi=150,
                         kde=False):
        if vector:
            Mx, My = timeseries
            M = np.sqrt(Mx**2 + My**2)
            fig = plt.figure(dpi=dpi, figsize=size)
            fig.suptitle('Histograms', fontsize=18)

            plt.subplot(1, 3, 1)
            sns.distplot(Mx, kde=kde)
            plt.title('$M_{x}$', fontsize=18)

            plt.subplot(1, 3, 2)
            sns.distplot(My, kde=kde)
            plt.title('$M_{y}$', fontsize=18)

            plt.subplot(1, 3, 3)
            sns.distplot(M, kde=kde)
            plt.title('|M|', fontsize=18)

        else:
            Mx = timeseries[0]
            fig = plt.figure(dpi=dpi, figsize=(10,5))
            plt.suptitle('Histogram', fontsize=18)

            plt.subplot(1, 2, 1)
            sns.distplot(Mx, kde=kde)
            plt.title('M', fontsize=18)

            plt.subplot(1, 2, 2)
            sns.distplot(np.sqrt(Mx**2), kde=kde)
            plt.title("|M|", fontsize=18)

        return fig

    def _slider(self, slider_data, init_pos=14, prefix='dt'):
        if prefix == 'Dt':
            t = 'Drift'
            t_tex = "\Delta t"
            sub_titles = ('driftX', 'driftY')
            scene1 = dict(
                xaxis=dict(showbackground=True),
                yaxis=dict(showbackground=True),
                zaxis=dict(showbackground=True, ),
                xaxis_title=r'mx',
                yaxis_title=r'my',
                zaxis_title=r'A1',
            )
            scene2 = dict(
                xaxis=dict(showbackground=True),
                yaxis=dict(showbackground=True),
                zaxis=dict(showbackground=True, ),
                xaxis_title=r'mx',
                yaxis_title=r'my',
                zaxis_title=r'A2',
            )
        else:
            t = 'Diff'
            t_tex = "\delta t"
            sub_titles = ('diffX', 'diffY')
            scene1 = dict(
                xaxis=dict(showbackground=True),
                yaxis=dict(showbackground=True),
                zaxis=dict(showbackground=True, ),
                xaxis_title=r'mx',
                yaxis_title=r'my',
                zaxis_title=r'B11',
            )
            scene2 = dict(
                xaxis=dict(showbackground=True),
                yaxis=dict(showbackground=True),
                zaxis=dict(showbackground=True, ),
                xaxis_title=r'mx',
                yaxis_title=r'my',
                zaxis_title=r'B22',
            )
        nrows, ncols = 1, 2
        title_template = r"$\text{{ {0} |  Auto correlation time : {1} }} | \text{{ Slider switched to }}{2}= {3}$"
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[
                [{
                    'type': 'scene'
                }, {
                    'type': 'scene'
                }],
            ],
            print_grid=False,
            subplot_titles=sub_titles,
            horizontal_spacing=0.1,
        )

        x, y = np.meshgrid(self.op_x, self.op_y)
        n = list(sub_titles)
        flag = 1
        for dt in slider_data:
            data = slider_data[dt]
            visible = 'legendonly'
            if flag:
                visible = True
            k = 0
            for r in range(1, nrows + 1):
                for c in range(1, ncols + 1):
                    marker_colour = 'blue'
                    if k % 2: marker_colour = 'red'
                    fig.append_trace(
                        go.Scatter3d(x=(x.flatten()),
                                     y=(y.flatten()),
                                     z=(data[k].flatten()),
                                     opacity=0.8,
                                     mode='markers',
                                     marker=dict(size=3, color=marker_colour),
                                     name="{}, {}".format(n[k], dt),
                                     visible=visible),
                        row=r,
                        col=c,
                    )
                    k = k + 1
            flag = 0

        fig.update_layout(
            autosize=True,
            scene_aspectmode='cube',
            scene1=scene1,
            scene2=scene2,
            #scene3 = scene,
            #scene4 = scene,
            title_text=title_template.format(t, self.autocorrelation_time,
                                             t_tex,
                                             list(slider_data.keys())[0]),
            height=600,
            width=900,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{
                                "type": ["scatter3d", "scatter3d"]
                            }],
                            #{'traces': [0, 1]}],
                            label="3D",
                            method="restyle"),
                        dict(
                            args=[{
                                "type": ["heatmap", "heatmap"]
                            }],
                            #{'traces': [0, 1]}],
                            label="Heatmap",
                            method="restyle")
                    ]),
                    pad={
                        "r": 10,
                        "t": 10
                    },
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"),
            ])

        dt_s = list(slider_data.keys())
        steps = []
        for i in range(len(slider_data)):
            step = dict(
                method='update',
                args=[{
                    "visible": ['legendonly'] * len(fig.data)
                }, {
                    "title":
                    title_template.format(t, self.autocorrelation_time, t_tex,
                                          str(dt_s[i])),
                }],  # layout attribute
                label='{} {}'.format(prefix,
                                     list(slider_data.keys())[i]))
            #step['args'][0][i*4:i*4+4] = [True for j in range(4)]
            step['args'][0]['visible'][i * 2:i * 2 +
                                       2] = [True for j in range(2)]
            steps.append(step)

        sliders = [
            dict(
                currentvalue={"prefix": "{} : ".format(prefix)},
                steps=steps,
            )
        ]

        fig.layout.sliders = sliders
        fig.layout.template = 'plotly_white'

        return fig

    def _thrace_pane(self, data):
        op_x = self.op_x.copy()
        op_y = self.op_y.copy()
        plane1 = []
        plane2 = []
        for y in data:
            nan_idx = np.where(np.isnan(y))
            try:
                p, x = self._fit_poly(op_x, y, deg=6)
                d = p(op_x)
            except Exception as e:
                d = np.zeros(y.shape)
            d[nan_idx] = np.nan
            plane1.append(d)

        for y in data.T:
            nan_idx = np.where(np.isnan(y))
            try:
                p, x = self._fit_poly(op_x, y, deg=6)
                d = p(op_x)
            except:
                d = np.zeros(y.shape)
            d[nan_idx] = np.nan
            plane2.append(d)

        plane1 = np.array(plane1)
        plane2 = np.array(plane2)
        err_1 = np.nanmean(np.sqrt(np.square(plane1 - data)))
        err_2 = np.nanmean(np.sqrt(np.square(plane2 - data.T)))
        if err_1 < err_2:
            return 0, plane1
        return 1, plane2

    def _plot_data(self,
                   data_in,
                   ax=None,
                   clear=True,
                   plot_plane=False,
                   title='title',
                   z_label='z',
                   label=None,
                   order=3,
                   m=False,
                   m_th=2,
                   dpi=150,
                   heatmap=False):
        fig = None
        if heatmap:
            return self._plot_heatmap(data_in, title=title)
        if ax is None:
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(projection="3d")
        data = data_in.copy()
        mask = np.where(((data > m_th * np.nanstd(data)) |
                         (data < -m_th * np.nanstd(data))))
        if m:
            #print(mask)
            data[mask] = np.nan
        if clear:
            ax.cla()
        op_x = self.op_x.copy()
        op_y = self.op_y.copy()
        plane = []
        if plot_plane:
            plane_id, plane = self._thrace_pane(data)

        x, y = np.meshgrid(op_x, op_y)
        z = data.copy()
        plt.suptitle(title, fontsize=16)

        ax.scatter3D(x, y, z.ravel(), label=label)
        if plot_plane:
            if plane_id:
                #print('Plane 2')
                ax.plot_surface(
                    y,
                    x,
                    plane,
                    rstride=1,
                    cstride=1,
                    alpha=0.5,
                )
            else:
                #print('Plane 1')
                ax.plot_surface(
                    x,
                    y,
                    plane,
                    rstride=1,
                    cstride=1,
                    alpha=0.5,
                )
        ax.set_xlabel('$m_x$', fontsize=16, labelpad=11)
        ax.set_ylabel('$m_y$', fontsize=16, labelpad=11)
        ax.set_zlabel(z_label, fontsize=16, labelpad=11)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        #Set ticks lable and its fontsize
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticks(np.linspace(-1, 1, 5))
        ax.set_yticks(np.linspace(-1, 1, 5))
        #plt.tight_layout()
        #plt.legend(prop={'size': 14})
        return fig, ax

    def _plot_heatmap(self, data, title='title', num_ticks=5):
        fig = plt.figure()
        plt.suptitle(title, verticalalignment='center', ha='right')
        ticks = self.op_x.copy()
        ticks_loc = np.linspace(0, len(ticks), num_ticks)
        ticks = np.linspace(min(ticks), max(ticks), num_ticks).round(2)
        ax = sns.heatmap(data,
                         xticklabels=ticks[::-1],
                         yticklabels=ticks,
                         cmap=plt.cm.coolwarm,
                         center=0)
        ax.set_xlabel('$m_x$', fontsize=16, labelpad=10)
        ax.set_ylabel('$m_y$', fontsize=16, labelpad=10)
        ax.set_xticks(ticks_loc)
        ax.set_yticks(ticks_loc)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig, None

    def _histogram3d(self,
                     x,
                     bins=20,
                     normed=False,
                     color='blue',
                     alpha=1,
                     hold=False,
                     plot_hist=False):
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
			
		Examples
		--------
		>>> r = np.random.randn(1000,2)
		>>> H, edges = np._histogram3d(r,bins=[10,15])
		"""

        if np.size(bins) == 1:
            bins = [bins, bins]

        if (len(x) == 2):
            x = x.T

        H, edges = np.histogramdd(x, bins, normed=normed)

        H = H.T
        X = np.array(
            list(np.linspace(min(edges[0]), max(edges[0]), bins[0])) * bins[1])
        Y = np.sort(
            list(np.linspace(min(edges[1]), max(edges[1]), bins[1])) * bins[0])

        dz = np.array([])

        for i in range(bins[1]):
            for j in range(bins[0]):
                dz = np.append(dz, H[i][j])

        Z = np.zeros(bins[0] * bins[1])

        dx = X[1] - X[0]
        dy = Y[bins[0]] - Y[0]

        if plot_hist:
            if (not hold):
                fig = plt.figure(dpi=300)
                ax = fig.add_subplot(111, projection='3d')
                colors = plt.cm.jet(dz.flatten() / float(dz.max()))
                ax.bar3d(X, Y, Z, dx, dy, dz, alpha=alpha, color=colors)
            else:
                try:
                    ax = plt.gca()
                    colors = plt.cm.jet(dz.flatten() / float(dz.max()))
                    ax.bar3d(X, Y, Z, dx, dy, dz, alpha=alpha, color=colors)
                except:
                    plt.close(plt.get_fignums()[-1])
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    colors = plt.cm.jet(dz.flatten() / float(dz.max()))
                    ax.bar3d(X, Y, Z, dx, dy, dz, alpha=alpha, color=colors)

        plt.xlabel('X')
        plt.ylabel('Y')
        edges = [X, Y]
        H = dz.reshape(bins[0], bins[1])

        #return H, edges;
        return H, edges, X, Y, Z, dx, dy, dz

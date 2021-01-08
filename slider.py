import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyFish
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import pickle

autocorr = pyFish.AutoCorrelation(fft=True)

def save_data(Mx, My, f_name, N=16, t_int=0.12):
	max_dt = autocorr._get_autocorr_time(Mx**2 + My**2)
	data_dict = dict()
	for n in tqdm(range(1, N+1), desc='Generating data'):
		if int((n/N)*max_dt) < 1:
			continue
		ch = pyFish.Characterize([Mx, My], t=None, t_int=t_int, dt=int((n/N)*max_dt))
		data_dict[int((n/N)*max_dt)] = ch.data()

	with open(f_name, 'wb') as handle:
		pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def slider(Mx, My, N=16, t_int=0.12, data_file=None):
	if data_file is None:
		max_dt = autocorr._get_autocorr_time(Mx**2 + My**2)
		data_dict = dict()
		for n in tqdm(range(1, N+1), desc='Generating data'):
			if int((n/N)*max_dt) < 1:
				continue
			ch = pyFish.Characterize([Mx, My], t=None, t_int=t_int, dt=int((n/N)*max_dt))
			data_dict[int((n/N)*max_dt)] = ch.data()

	else:
		with open(data_file, 'rb') as handle:
			data_dict = pickle.load(handle)
	op_x, op_y = np.arange(-1,1,0.1), np.arange(-1,1,0.1)
	x, y = np.meshgrid(op_x, op_y)
	n = ['driftX', 'driftY', 'diffX', 'diffY']
	fig = make_subplots(rows=2, cols=2,
					specs=[[{'is_3d': True}, {'is_3d': True}],
						   [{'is_3d': True}, {'is_3d': True}]],
					print_grid=False, subplot_titles=('driftX','driftY','diffX','diffY'),
				   horizontal_spacing=0.01,
				   vertical_spacing=0.05,)

	flag = 1
	for dt in data_dict:
		data = data_dict[dt]
		visible = 'legendonly'
		if flag:
			visible = True
		k = 0
		for r in range(1,3):
			for c in range(1,3):
				marker_colour = 'blue'
				if k%2: marker_colour = 'red'
				fig.append_trace(
					go.Scatter3d(
						x=(x.flatten()),
						y=(y.flatten()),
						z=(data[k].flatten()),
						opacity=0.8,
						mode='markers',
						marker=dict(size=3,color=marker_colour),
						name="{}, {}".format(n[k], dt),visible=visible),
					row=r, col=c,)
				k = k + 1
		flag = 0


	scene = dict(xaxis = dict(showbackground=True),
					yaxis = dict(showbackground=True),
					zaxis = dict(showbackground=True,),
					xaxis_title='mx',
					yaxis_title='my',
					zaxis_title='Z',)

	fig.update_layout(
		autosize=False,
		scene_aspectmode='cube',
		scene1 = scene,
		scene2 = scene,
		scene3 = scene,
		scene4 = scene,
		title_text='3D subplots',
		height=1000,
		width=1000,
	)

	dt_s = list(data_dict.keys())
	steps = []
	for i in range(len(data_dict)):
		step = dict(
			method = 'update',  
			#args = ['visible', ['legendonly'] * len(fig.data),],
			args=[{"visible": [False] * len(fig.data)},
				  {"title": "Slider switched to dt = " + str(dt_s[i])}],  # layout attribute
		)
		#step['args'][0][i*4:i*4+4] = [True for j in range(4)]
		step['args'][0]['visible'][i*4:i*4+4] = [True for j in range(4)]
		steps.append(step)

	sliders = [dict(
		currentvalue={"prefix": "dt: "},
		steps = steps,
	)]

	fig.layout.sliders = sliders

	return fig


Mx_2, My_2 = np.loadtxt('syn_pairwise.txt').T
Mx_3, My_3 = np.loadtxt('syn_ternary.txt').T

#save_data(Mx_2, My_2, 'slider_data_parwise.pkl')
#save_data(Mx_3, My_3, 'slider_data_ternary.pkl')
s = slider(Mx=None, My=None, data_file='slider_data_parwise.pkl')
s.show()
s2 = slider(Mx=None, My=None, data_file='slider_data_ternary.pkl')
s2.show()
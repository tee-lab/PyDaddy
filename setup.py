from setuptools import setup, find_packages

with open("README.md", 'r') as f:
		long_description = f.read()

with open('requirements.txt', 'r') as f:
	INSTALL_REQUIREMENTS = f.read().split()
#INSTALL_REQUIREMENTS.append('pylustrator@git+https://github.com/ashwinkk23/pylustrator@load_matplotlib_figure#egg=pylustrator-1.0.0')

setup(
	name='pydaddy',
	version='0.1.5',
	description='Package to analyse stochastic time series data',
	long_description=long_description,
	long_description_content_type='text/markdown',
	author='Ashwin Karichannavar',
	author_email='ashwinkk.23@gmail.com',
	url="https://github.com/tee-lab/pydaddy",
	packages=find_packages(),
	classifiers=[
		# How mature is this project? Common values are
		#   3 - Alpha
		#   4 - Beta
		#   5 - Production/Stable
		'Development Status :: 4 - Beta',
		'Environment :: Console',
		'Intended Audience :: Science/Research',
		'Intended Audience :: Education',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Operating System :: Microsoft :: Windows :: Windows 10',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Unix',
		'Programming Language :: Python :: Implementation',
		'Programming Language :: Python :: 3 :: Only',
		'Topic :: Scientific/Engineering :: Information Analysis',
		'Topic :: Scientific/Engineering :: Visualization',
		'Topic :: Scientific/Engineering :: Mathematics',
		'Topic :: Scientific/Engineering :: Physics',
		'Topic :: Documentation :: Sphinx',
	],
	include_package_data=True,
	package_data={'': ['data/fish_data/*.csv', 'data/model_data/scalar/*.csv', 'data/model_data/vector/*.csv', 'report/report']},
	install_requires=INSTALL_REQUIREMENTS, #external packages as dependencies
	entry_points={'console_scripts': [
			'pydaddy=pydaddy.__console__:main',
		]
	}
)

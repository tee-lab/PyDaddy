from setuptools import setup, find_packages

with open("README.md", 'r') as f:
		long_description = f.read()

with open('requirements.txt', 'r') as f:
	INSTALL_REQUIREMENTS = f.read().split()
	

setup(
	name='pyFish',
	version='1.17',
	description='Package to analyse stochastic time series data',
	long_description=long_description,
	long_description_content_type='text/markdown',
	author='Ashwin Karichannavar',
	author_email='ashwinkk.23@gmail.com',
	url="https://github.com/tee-lab/pyFish",
	packages=find_packages(),
	classifiers=[
		# How mature is this project? Common values are
		#   3 - Alpha
		#   4 - Beta
		#   5 - Production/Stable
		'Development Status :: 4 - Beta',

		'Intended Audience :: Developers, Researchers',
		'Topic :: Data Analysis :: Stochastic Timeseries',

		'License :: OSI Approved :: GNU General Public License v3.0',

		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
		'Programming Language :: Python :: 3 :: Only',
	],
	include_package_data=True,
	package_data={'': ['data/fish_data/*.csv', 'data/model_data/scalar/*.csv', 'data/model_data/vector/*.csv']},
	install_requires=INSTALL_REQUIREMENTS, #external packages as dependencies
	entry_points={'console_scripts': [
			'pyFish=pyFish.__console__:main',
		]
	}
)
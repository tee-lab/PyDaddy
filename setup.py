from setuptools import setup, find_packages

with open("README.md", 'r') as f:
		long_description = f.read()

with open('requirements.txt', 'r') as f:
	INSTALL_REQUIREMENTS = f.read().split()
	

setup(
	name='pyFish',
	version='1.0',
	description='Package to analyse stochastic time series data',
	license="",
	long_description=long_description,
	author='Ashwin Karichannavar',
	author_email='ashwinkk.23@gmail.com',
	url="https://github.com/ashwinkk23/pyFish",
	packages=find_packages(),  #same as name
	include_package_data=True,
	package_data={'': ['data/extras/*.csv', 'data/pairwise/*.csv', 'data/ternary/*.csv', 'data/vector/*.csv']},
	install_requires=INSTALL_REQUIREMENTS, #external packages as dependencies
	entry_points={'console_scripts': [
			'pyFish=pyFish.__console__:main',
		]
	}
)
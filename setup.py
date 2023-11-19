from setuptools import setup, find_packages

with open('README.md') as f:
    readme_file = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

setup(
    name='smartsod2d',
    version='1.0',
    description='A ML framework for the SOD2D CFD solver using SmartSim',
    long_description=readme_file,
    author='Bernat Font, Francisco Alcántara Ávila',
    author_email='bernatfontgarcia@gmail.com',
    keywords=['reinforcement learning, computational fluid dynamics'],
    url='https://github.com/b-fg/smartsod2d',
    license=license_file,
    packages=find_packages(exclude=('tests', 'docs')),
		setup_requires=['numpy'],
		install_requires=['numpy', 'smartsim==0.4.2', 'smartredis', 'tensorflow', 'tf_agents==0.10.0', 'matplotlib']
)

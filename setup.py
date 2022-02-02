from setuptools import find_packages, setup

setup(
      name='empylib',
      version='0.1.0',
      description='Standard python library for computational electromagnetics',
      author='Francisco V. Ramirez-Cuevas',
      author_author_email='francisco.ramirez.c@uai.cl',
      url='',
      packages=find_packages(exclude=()),
      install_requires=[
          'numpy', 'os', 'scipy'
          ]
      )
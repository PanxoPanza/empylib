from setuptools import find_packages, setup

setup(
      name='empylib',
      version='0.0.1',
      description='Standard python library for computational electromagnetics',
      long_description=open('README.md').read() + '\n\n' + open(CHANGELOG.txt).read(),
      author='Francisco V. Ramirez-Cuevas',
      author_author_email='francisco.ramirez.c@uai.cl',
      url='https://github.com/PanxoPanza/empylib.git',
      packages=find_packages(),
      licence='MIT',
      keywords='electromagnetics',
      #install_requires=[
      #    'numpy', 'os', 'scipy'
      #    ],
      classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
      python_requires='>=3.6, <4',
      )
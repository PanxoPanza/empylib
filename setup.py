from setuptools import find_packages, setup
setup(
      name='empylib',
      version='0.1.2',
      description='Standard python library for computational electromagnetism',
      author='Francisco V. Ramirez-Cuevas',
      author_email='fvr@alumni.cmu.edu',
      url='https://github.com/PanxoPanza/empylib.git',
      packages=['empylib'],
      license='MIT',
      keywords='electromagnetism',
      install_requires=['numpy', 
                        'scipy', 
                        'iadpython'
                        ],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
      python_requires='>=3.6, <4',
      )
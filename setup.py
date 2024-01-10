#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from distutils.core import setup
from setuptools import find_packages

package_name = 'empylib' 
PACKAGES = [package_name]

def get_init_val(val, packages=PACKAGES):
    pkg_init = "%s/__init__.py" % PACKAGES[0]
    value = '__%s__' % val
    fn = open(pkg_init)
    for line in fn.readlines():
        if line.startswith(value):
            return line.split('=')[1].strip().strip("'")

setup(
    name=get_init_val('title'),
    version=get_init_val('version'),
    description=get_init_val('description'),
    long_description=open('README.md').read(),
    author=get_init_val('author'),
    url=get_init_val('url'),
    package_data={'': ['LICENSE', 'NOTICE', '*.nk', '*.txt']},
    license=get_init_val('license'),
    keywords='electromagnetism',
    install_requires=['numpy',
                      'scipy',
                      'iadpython',
                      'refidx',
                      'pandas'
                      ],
    packages=find_packages(include=[package_name, package_name + '.*']),
)
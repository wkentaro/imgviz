#!/usr/bin/env python

from __future__ import print_function

import distutils.spawn
from setuptools import find_packages
from setuptools import setup
import shlex
import subprocess
import sys


version = '0.4.0'


if sys.argv[1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git submodule update github2pypi',
        'git pull origin master',
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/imgviz-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def get_install_requires():
    install_requires = []
    with open('requirements.txt') as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()

    try:
        import github2pypi
    except ImportError:
        return long_description

    return github2pypi.replace_url(
        slug='wkentaro/imgviz', content=long_description
    )


setup(
    name='imgviz',
    version=version,
    packages=find_packages(exclude=['github2pypi']),
    install_requires=get_install_requires(),
    extras_require={
        'all': ['scikit-image', 'scikit-learn'],
    },
    description='Image Visualization Tools',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    package_data={'imgviz': ['data/*']},
    include_package_data=True,
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='http://github.com/wkentaro/imgviz',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)

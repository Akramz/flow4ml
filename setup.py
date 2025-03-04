#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Akram Zaytar",
    author_email='akramzaytar@microsoft.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="The goal of this repo is to iterate our way to the best model for building density detection on a global scale.",
    entry_points={
        'console_scripts': [
            'come_and_train=come_and_train.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='come_and_train',
    name='come_and_train',
    packages=find_packages(include=['come_and_train', 'come_and_train.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/akramz/come_and_train',
    version='0.1.0',
    zip_safe=False,
)

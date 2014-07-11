from setuptools import setup, find_packages
from os.path import join, dirname, basename, splitext
from glob import glob


def read(name):
    return open(join(dirname(__file__), name)).read()


setup(
    name='pyphase',
    version='0.0.1',
    license='MIT',

    description='Methods of calculating relative phase and related algorithms, implemented in Python.',
    long_description=read('README.rst'),

    author='Henry S. Harrison',
    author_email='henry.schafer.harrison@gmail.com',

    url='https://bitbucket.org/hharrison/pyphase',
    download_url='https://bitbucket.org/hharrison/pyphase/get/default.tar.gz',

    package_dir={'': 'src'},
    packages=find_packages('src'),

    keywords='phase signal oscillation oscillators periodic peaks hkb',

    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
    ],

    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
)

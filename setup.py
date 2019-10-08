from distutils.core import setup

from setuptools import find_packages

setup(
    name='chainer-dataset-utility',
    version='0.1.0',
    description='',
    author='higumachan',
    author_email='yuta.hinokuma725@gmail.com',
    license='MIT',
    url='https://github.com/higumachan/chainer-dataset-utility',
    packages=find_packages(exclude=('tests')),
)
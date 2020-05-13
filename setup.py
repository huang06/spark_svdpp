from setuptools import setup, find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spark-svdpp',
    version='0.1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages()
)

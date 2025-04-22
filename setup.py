# setup.py
from setuptools import setup, find_packages

setup(
    name="newslens_ai",
    version="0.1",
    packages=find_packages(),  # this will pick up ner/ and scrapers/
)
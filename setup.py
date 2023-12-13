import os
from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        fcontent = f.read()
    return fcontent
    

setup(
    name = "llmcal",
    version = "0.0.1",
    author = "Lautaro Estienne",
    author_email = "lestienne@fi.uba.ar",
    description = ("Code to analyze calibration of LLMs used in classification tasks"),
    keywords = "calibration UCPA SUCPA logistic regression psr mahalanobis",
    url = "https://github.com/LautaroEst/llmcal",
    packages=['llmcal'],
    long_description=read('Readme.md')
)
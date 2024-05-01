from setuptools import setup

setup(
    name='llmcal',
    version='0.1.0',
    packages=['llmcal'],
    install_requires=[
        'fire',
        'lightning',
        'transformers[sentencepiece]',
        'datasets',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'tensorboard',
        'jupyter',
        'lit_gpt[all]',
        'deepspeed'
    ],
)
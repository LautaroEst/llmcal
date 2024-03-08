from setuptools import setup

setup(
    name='llmcal',
    version='0.1.0',
    packages=['llmcal'],
    install_requires=[
        'fire',
        'bitsandbytes==0.41.0',
        'lightning',
        'transformers[sentencepiece]',
        'datasets',
        'scikit-learn',
        'pandas',
        'tensorboard',
        'lit_gpt @ git+https://github.com/Lightning-AI/lit-gpt.git',
    ],
)
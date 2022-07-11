from setuptools import setup, find_packages

setup(
    name='cogs',
    version='1.0.0',
    description='CoGS: Controllable Generation and Search from Sketch and Style',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)

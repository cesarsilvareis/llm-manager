from setuptools import setup, find_packages

setup(
    name='experiment-medical-llms',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'medllmexp=src.main:main',
        ],
    },
)

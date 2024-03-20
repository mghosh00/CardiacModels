#
# cardiac_models setuptools script
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the neural_network module.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('cardiac_models'))
    version = "1.0.0"
    sys.path.pop()

    return version


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as file:
        for line in file:
            requirements.append(line)
    return requirements


setup(
    # Module name
    name='cardiac_models',

    # Version
    version=get_version(),

    description='An exploration of cardiac models',

    maintainer='Matthew Ghosh',

    maintainer_email='matthew.ghosh@gtc.ox.ac.uk',

    url='https://github.com/mghosh00/CardiacModels',

    # Packages to include
    packages=find_packages(include=('cardiac_models', 'cardiac_models.*')),

    # List of dependencies
    install_requires=get_requirements(),

    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',
        ],
        'dev': [
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)

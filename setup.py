# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
import os
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "sisyphe", "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="sisyphe",
    version="0.1",
    description="Simulation of Systems of Particles with High Efficiency",  # Required
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="sisyphe.readthedocs.io",
    project_urls={
        "Bug Reports": "https://github.com/antoinediez/Sisyphe/issues",
        "Source": "https://github.com/antoinediez/Sisyphe",
    },
    author="Antoine Diez",
    author_email="antoine.diez18@imperial.ac.uk",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="particles gpu",
    packages=[
        "sisyphe",
    ],
    package_data={
        "sisyphe": [
            "readme.md",
            "licence.txt",
        ]
    },
    install_requires=[
        "numpy",
        "torch",
    ],
    extras_require={
        "full": [
            "pykeops",
        ],
    },
)
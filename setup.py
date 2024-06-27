# Standard Libraries
from setuptools import setup, find_namespace_packages
import pathlib

APP_NAME = "ecoevocrm"
VERSION = "1.0.1"
LICENSE = "MIT"
AUTHOR = "Ryan Seamus McGee"
DESCRIPTION = "Computational frameworks for extended consumer resource models of eco-evoutionary community dynamics."
URL = "https://github.com/tikhonov-group/ecoevocrm.git"

# Directory containing this file
HERE = pathlib.Path(__file__).parent

# Text of README file
README = (HERE / "README.md").read_text()

setup(
    name=APP_NAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "argparse"
    ],
    extra_require={

    },
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.yaml", "*.json"]},
    entry_points="""
    [console_scripts]
    """,
    python_requires=">=2.6",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)

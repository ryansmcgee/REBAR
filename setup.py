# Standard Libraries
from setuptools import setup, find_namespace_packages
import pathlib

APP_NAME = "rebar-py"
VERSION = "1.0.2"
LICENSE = "MIT"
AUTHOR = "Ryan Seamus McGee"
DESCRIPTION = "REBAR (Removing the Effects of Bias through Analysis of Residuals): Improving the accuracy of bulk fitness assays by correcting barcode processing biases."
URL = "https://github.com/ryansmcgee/REBAR.git"

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
        "pandas",
        "json",
        "warnings"
    ],
    extra_require={

    },
    packages=find_namespace_packages(include=["src"], exclude=['src.REBAR.chen-case-study', 'src.REBAR.kinsler-case-study', 'src.REBAR.sensitivity-analyses']),
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

"""
Setup configuration for Vision2Slope package.
"""

import re
from pathlib import Path

from setuptools import setup, find_packages

HERE = Path(__file__).parent

# Read paths relative to this file so setup works from any working directory
with open(HERE / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Single source of truth for the version: vision2slope/__init__.py
version = re.search(
    r'^__version__ = "([^"]+)"',
    (HERE / "vision2slope" / "__init__.py").read_text(encoding="utf-8"),
    re.MULTILINE
).group(1)

setup(
    name="vision2slope",
    version=version,
    author="Cubics Yang",
    author_email="yangchen@nnu.edu.cn",
    description="Integrated pipeline for road slope analysis from street view images",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    # zensvi requires Python >=3.9,<3.13
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "gsv": ["streetlevel"],
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "vision2slope=vision2slope.cli:main",
        ],
    },
)

"""
Setup configuration for Vision2Slope package.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vision2slope",
    version="1.0.0",
    author="Cubics Yang",
    author_email="yangchen@nnu.edu.cn",
    description="Integrated pipeline for road slope analysis from street view images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CubicsYang/Vision2Slope",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vision2slope=vision2slope.cli:main",
        ],
    },
)

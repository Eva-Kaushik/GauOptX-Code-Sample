import os
from setuptools import setup, find_packages

def read(file_name):
    """Helper function to read files relative to the setup script location."""
    with open(os.path.join(os.path.dirname(__file__), file_name), encoding="utf-8") as f:
        return f.read()

# Define the current version of the package
__version__ = "2.0.0"

# Define the required dependencies
REQUIRED_PACKAGES = [
    "numpy>=1.21",
    "scipy>=1.7",
    "matplotlib>=3.4",
    "GPy>=1.10"
]

# Define optional dependencies
EXTRAS_REQUIRE = {
    "optimizer": ["DIRECT", "cma", "pyDOE", "sobol_seq", "emcee"],
    "docs": ["Sphinx>=5.0", "sphinx_rtd_theme", "jupyter", "IPython"],
    "dev": ["pytest", "flake8", "black"],
}

# Setup configuration
setup(
    name="GauOptX",
    version=__version__,
    description="A Flexible Bayesian Optimization Framework for Machine Learning and Experimentation",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    keywords=[
        "bayesian-optimization",
        "machine-learning",
        "gaussian-processes",
        "hyperparameter-tuning",
        "optimization",
    ],
    url="https://github.com/YourUsername/GauOptX",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    project_urls={
        "Source Code": "https://github.com/YourUsername/GauOptX",
        "Issue Tracker": "https://github.com/YourUsername/GauOptX/issues",
    },
    entry_points={
        "console_scripts": [
            "gauoptx=gauoptx.cli:main",
        ]
    },
)

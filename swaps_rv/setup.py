"""
Legacy (optional) setup.py for people who still run
   $ pip install .
instead of the modern PEP-517 build.

You may delete this file if your workflows always use
   $ pip install --config-settings editable=true .
or `pipx`, `uv`, `hatch`, etc.; the *pyproject.toml* alone is enough.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    # ----------------------------- core metadata (mirrors pyproject.toml)
    name="swaps-rv",                      # distribution name (hyphen OK)
    version="0.1.0",
    description=(
        "Tiered Gaussian-Process + ANN framework for fixed-income relative-value research"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your-Desk-Quant-Team",
    license="MIT",
    python_requires=">=3.11",            # keep consistent with pyproject

    # ----------------------------- package discovery
    package_dir={"": "src"},             # root import namespace maps to ./src
    packages=find_packages(where="src", exclude=("tests", "notebooks")),
    include_package_data=True,

    # ----------------------------- runtime dependencies
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.11",
        "numba>=0.59",
        "jax>=0.4",
        "jaxlib>=0.4",
        "dm-haiku>=0.0.11",
        "optax>=0.1.7",
        "matplotlib>=3.8",
        "pandas>=2.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "ruff>=0.3",
            "black>=24.3",
            "ipython",
        ],
    },

    # ----------------------------- Trove classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
    ],

    zip_safe=False,
)

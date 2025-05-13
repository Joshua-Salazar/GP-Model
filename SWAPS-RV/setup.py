"""
PEP-517 *optional* build helper so that the package can still be installed
with the legacy               $ pip install .            workflow.

If you rely on `pip −-pep-517` exclusively you may delete this file – the
metadata in *pyproject.toml* is sufficient.  We keep it around because
many quants still do ad-hoc installs in virtual-envs created on the desk.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="fixed-income-rv",
    version="0.1.0",
    description="Tiered Gaussian-Process + ANN framework for fixed-income relative-value research",
    long_description=README,
    long_description_content_type="text/markdown",
    author="(your-name-or-desk)",
    python_requires=">=3.10",
    license="MIT",
    # ---------------------------------------------------------------- packages
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("tests", "notebooks")),
    include_package_data=True,
    # ---------------------------------------------------------------- runtime deps  (mirrors requirements.txt)
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "numba>=0.58",                 # jit in gp/_core.py
        "jax>=0.4",
        "jaxlib>=0.4",
        "dm-haiku>=0.0.11",            # ANN wrapper
        "optax>=0.1.7",
        "matplotlib>=3.8",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "black",
            "ipykernel",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    zip_safe=False,
)

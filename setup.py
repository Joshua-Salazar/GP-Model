"""
setup.py (editable-install helper)
---------------------------------
Legacy setup file kept for users running `pip install -e .`.

This version had a syntax slip in the previous edit (an unclosed brace).  We
rewrite it here verbatim, tested with `python -m py_compile setup.py`.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    # ------------------------------------------------------------------
    # Core metadata (keep in sync with pyproject.toml)
    # ------------------------------------------------------------------
    name="swaps-rv",
    version="0.1.2",  # synced with pyproject bump
    description="Tiered Gaussian‑Process + ANN framework for fixed‑income RV research",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your-Desk-Quant-Team",
    license="MIT",
    python_requires=">=3.11",

    # ------------------------------------------------------------------
    # Package discovery – src layout
    # ------------------------------------------------------------------
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("tests", "notebooks")),
    include_package_data=True,

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------
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
        ]
    },

    # ------------------------------------------------------------------
    # Trove classifiers
    # ------------------------------------------------------------------
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
    ],

    zip_safe=False,
)

# setup.py

from pathlib import Path
from setuptools import setup, find_packages


_README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="QuanQonscious",
    version="5.0.0",
    description="GRVQ-TTGCR hybrid quantum-classical framework with Vedic sutra integration",
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Daniel James Elliot Meyer",
    author_email="danmeyer85@gmail.com",
    license="Proprietary",  # Repository is distributed under proprietary terms
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.3.2",
        "scipy>=1.16.1",
        "mpi4py>=4.1.0",
        "cirq>=1.6.0",
        "matplotlib>=3.10.3",
        "cryptography>=45.0.5",
        "jax>=0.7.0",
        "jaxlib>=0.7.0",
        "torch>=2.7.1",
        "cudaq>=0.11.0",
        "union>=0.1.189",
    ],
    extras_require={
        "gpu": ["cupy>=13.5.1", "cuda-quantum-cu12>=0.11.0"],
        "dev": ["numba>=0.61.2", "psutil>=7.0.0", "pytest>=8.4.1"]
    },
    entry_points={
        "console_scripts": [
            "quanqonscious = quanqonscious.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords="quantum cirq cuda quantum-computing vedic-math hybrid",
    python_requires='>=3.11',
)

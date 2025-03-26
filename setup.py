# setup.py

from setuptools import setup, find_packages

setup(
    name="QuanQonscious",
    version="0.1.0",
    description="GRVQ-TTGCR hybrid quantum-classical framework with Vedic sutra integration",
    author="Your Name",
    author_email="your.email@example.com",
    license="Proprietary",  # Assuming proprietary, adjust if open-source
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "cirq>=0.14.0",
        "mpi4py>=3.0.0",
    ],
    extras_require={
        "gpu": ["cupy>=9.0.0", "cuda-quantum-cu12>=0.1.0"],
        "dev": ["numba>=0.55", "psutil>=5.8"]
    },
    entry_points={
        "console_scripts": [
            "quanqonscious = QuanQonscious.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords="quantum cirq cuda quantum-computing vedic-math hybrid",
    python_requires='>=3.8',
)

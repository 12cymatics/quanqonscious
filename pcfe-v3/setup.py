from setuptools import setup, find_packages

setup(
    name="pcfe-v3",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "mpi4py",#!/usr/bin/env python3
"""
Setup script for Proto-Consciousness Field Engine (PCFE) v3.0
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Check Python version
if sys.version_info < (3, 10):
    print("Error: PCFE requires Python 3.10 or later")
    sys.exit(1)

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "docs" / "README.md").read_text(encoding="utf-8")

# Core requirements
install_requires = [
    # Core scientific computing
    "numpy>=1.24.3",
    "scipy>=1.11.3",
    "torch>=2.1.0",
    "matplotlib>=3.7.2",
    "seaborn>=0.12.2",
    
    # Quantum computing
    # Note: cuda-quantum and qiskit need separate installation
    "cirq>=1.2.0",
    
    # GPU computing
    # Note: cupy needs CUDA-specific version
    "numba>=0.58.0",
    
    # MPI
    "mpi4py>=3.1.4",
    
    # Data handling
    "h5py>=3.9.0",
    "zarr>=2.16.1",
    "pandas>=2.0.3",
    
    # Visualization
    "plotly>=5.17.0",
    "vispy>=0.13.0",
    "pyvista>=0.42.2",
    
    # ML/Analysis
    "scikit-learn>=1.3.1",
    "networkx>=3.1",
    "tensornetwork>=0.4.6",
    "opt-einsum>=3.3.0",
    
    # Utilities
    "pyyaml>=6.0.1",
    "dill>=0.3.7",
    "aiofiles>=23.2.1",
    "psutil>=5.9.5",
    "GPUtil>=1.4.0",
    
    # Testing
    "pytest>=7.4.2",
    "hypothesis>=6.87.0",
    
    # Deployment
    "docker>=6.1.3",
    "kubernetes>=28.1.0",
]

# Optional dependencies
extras_require = {
    'cuda11': ['cupy-cuda11x>=12.2.0'],
    'cuda12': ['cupy-cuda12x>=12.2.0'],
    'quantum': ['cuda-quantum==0.6.0', 'qiskit>=0.45.0'],
    'viz': ['vtk>=9.2.6', 'mayavi>=4.8.1', 'napari>=0.4.18'],
    'dev': [
        'black>=23.9.1',
        'flake8>=6.1.0',
        'mypy>=1.5.1',
        'pre-commit>=3.4.0',
        'sphinx>=7.2.6',
        'sphinx-rtd-theme>=1.3.0',
    ],
    'all': [],  # Will be populated below
}

# Combine all extras
all_extras = []
for extra in extras_require.values():
    if isinstance(extra, list):
        all_extras.extend(extra)
extras_require['all'] = list(set(all_extras))

setup(
    name="pcfe",
    version="3.0.0",
    author="Proto-Consciousness Field Engine Team",
    author_email="pcfe-dev@example.com",
    description="Proto-Consciousness Field Engine - Hybrid quantum-classical simulation of emergent consciousness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/pcfe-v3",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/pcfe-v3/issues",
        "Documentation": "https://pcfe.readthedocs.io",
        "Source Code": "https://github.com/YOUR_USERNAME/pcfe-v3",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Include additional files
    include_package_data=True,
    package_data={
        "pcfe": [
            "config/*.yaml",
            "docker/Dockerfile",
            "docker/requirements.txt",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "pcfe=pcfe_final_integration:main",
            "pcfe-test=pcfe_final_integration:quick_test_run",
            "pcfe-benchmark=pcfe_final_integration:benchmark_run",
            "pcfe-validate=pcfe_validation_deployment:run_all_validations",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Operating System :: POSIX :: Linux",
        "Framework :: Pytest",
    ],
    
    keywords=[
        "quantum computing",
        "consciousness",
        "physics simulation", 
        "cuda",
        "mpi",
        "scientific computing",
        "emergence",
        "complex systems",
    ],
)

# Post-installation message
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  PCFE v3.0 Installation Complete!                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Next steps:

1. Install CUDA-specific packages:
   - For CUDA 11.x: pip install cupy-cuda11x cuda-quantum
   - For CUDA 12.x: pip install cupy-cuda12x cuda-quantum

2. Verify installation:
   pcfe --test

3. Run a quick simulation:
   pcfe --mode performance --iterations 1000

4. For distributed runs:
   mpirun -n 4 pcfe --distributed

Documentation: https://github.com/YOUR_USERNAME/pcfe-v3
""")
        "matplotlib"
    ],
)

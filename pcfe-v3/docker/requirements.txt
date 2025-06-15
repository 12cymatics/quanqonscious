# Core scientific computing
numpy==1.24.3
scipy==1.11.3
torch==2.1.0+cu118  # CUDA 11.8 version
matplotlib==3.7.2
seaborn==0.12.2

# GPU computing
cupy-cuda11x==12.2.0
numba==0.58.0

# Quantum computing
cuda-quantum==0.6.0
cirq==1.2.0
# Note: Qiskit removed - using CUDA-Quantum exclusively

# MPI and distributed
mpi4py==3.1.4

# Data handling
h5py==3.9.0
zarr==2.16.1
pandas>=2.0.3

# Visualization
plotly==5.17.0
vispy==0.13.0
vtk==9.2.6
pyvista==0.42.2

# ML/Analysis
scikit-learn==1.3.1
networkx==3.1
tensornetwork==0.4.6
opt-einsum==3.3.0

# Utilities
pyyaml==6.0.1
dill==0.3.7
aiofiles==23.2.1
psutil==5.9.5
GPUtil==1.4.0

# Testing
pytest==7.4.2
hypothesis==6.87.0

# Deployment
docker==6.1.3
kubernetes==28.1.0

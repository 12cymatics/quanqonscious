# PCFE v3.0 Production Container - CUDA-Quantum Edition
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    build-essential \
    cmake \
    mpich \
    libmpich-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY docker/requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install CUDA-Quantum (requires specific installation)
RUN pip3 install cuda-quantum==0.6.0 --extra-index-url https://pypi.nvidia.com

# Install CuPy for CUDA 11.8
RUN pip3 install cupy-cuda11x

# Copy source code
COPY src/ /app/src/
COPY config/ /app/config/
COPY examples/ /app/examples/

# Set up directories
RUN mkdir -p /app/checkpoints /app/logs /app/results

# Environment for MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# CUDA-Quantum environment
ENV CUDAQ_DEFAULT_TARGET=nvidia

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Default command
CMD ["python3", "src/pcfe_final_integration.py", "--config", "config/default_confi

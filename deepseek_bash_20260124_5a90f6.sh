%%bash
# Install EVERY dependency with FULL versions
apt-get update && apt-get upgrade -y

# CORE COMPILATION TOOLS
apt-get install -y \
    build-essential=12.8* \
    cmake=3.22* \
    git=1:2.34* \
    ninja-build=1.10* \
    pkg-config=0.29* \
    autoconf=2.71* \
    automake=1:1.16* \
    libtool=2.4.* \
    gdb=11.1* \
    valgrind=1:3.18*

# BOOST COMPLETE SUITE
apt-get install -y \
    libboost-all-dev=1.74.0* \
    libboost-serialization-dev=1.74.0* \
    libboost-multiprecision-dev=1.74.0* \
    libboost-system-dev=1.74.0* \
    libboost-filesystem-dev=1.74.0* \
    libboost-thread-dev=1.74.0* \
    libboost-program-options-dev=1.74.0* \
    libboost-test-dev=1.74.0* \
    libboost-chrono-dev=1.74.0* \
    libboost-date-time-dev=1.74.0* \
    libboost-atomic-dev=1.74.0* \
    libboost-timer-dev=1.74.0*

# EIGEN & LINEAR ALGEBRA
apt-get install -y \
    libeigen3-dev=3.4.0* \
    liblapack-dev=3.10* \
    libblas-dev=3.10* \
    libopenblas-dev=0.3.20* \
    libsuitesparse-dev=1:5.10* \
    libarpack2-dev=1:3.8.0*

# PARALLEL COMPUTING
apt-get install -y \
    ocl-icd-opencl-dev=2.2.14* \
    opencl-headers=3.0* \
    libomp-dev=1:12.0* \
    libtbb-dev=2021.5* \
    libfftw3-dev=3.3.10* \
    libhdf5-dev=1.10.7* \
    libnetcdf-dev=1:4.8.1* \
    libopenmpi-dev=4.1.2*

# TESTING FRAMEWORKS
apt-get install -y \
    libgtest-dev=1.11.0* \
    libgmock-dev=1.11.0* \
    python3-gtest=1.11.0*

# VISUALIZATION & UTILITIES
apt-get install -y \
    graphviz=2.50.0* \
    gnuplot=5.4.3* \
    texlive-latex-extra=2021.20220204* \
    doxygen=1.9.1* \
    plantuml=1:1.2022.3* \
    libxml2-dev=2.9.13* \
    libxslt1-dev=1.1.35* \
    libssl-dev=1.1.1* \
    libbz2-dev=1.0.8* \
    liblzma-dev=5.2.5* \
    zlib1g-dev=1:1.2.11*

# PYTHON BINDINGS
pip install --upgrade \
    pybind11[global]==2.10.4 \
    numpy==1.24.0 \
    scipy==1.10.0 \
    matplotlib==3.7.0 \
    pandas==2.0.0 \
    jupyter==1.0.0 \
    networkx==3.0 \
    plotly==5.14.0 \
    ipywidgets==8.0.6

# INSTALL CUDA 12.0 COMPLETE
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y \
    cuda-toolkit-12-0 \
    cuda-libraries-dev-12-0 \
    cuda-driver-dev-12-0 \
    libcublas-dev-12-0 \
    libcudnn8-dev=8.9.2.* \
    libnccl-dev=2.18.*

# SETUP CUDA ENVIRONMENT
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.0' >> ~/.bashrc
source ~/.bashrc

# VERIFY INSTALLATIONS
echo "=== VERIFICATION ==="
nvcc --version
cmake --version
g++ --version
python3 -c "import pybind11; print(f'PyBind11: {pybind11.__version__}')"

# BUILD AND INSTALL GOOGLE TEST
cd /usr/src/gtest
sudo cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .
sudo make -j$(nproc)
sudo cp lib/*.so /usr/lib
sudo ldconfig

echo "=== DEPENDENCY INSTALLATION COMPLETE ==="
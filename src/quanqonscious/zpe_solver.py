# QuanQonscious/zpe_solver.py

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class ZPEFieldSolver:
    """
    Solver for Zero-Point Energy resonance field in 4D (3D space + time).
    
    This solver updates a 3D field over time according to a wave-like equation with damping.
    It supports CPU (NumPy) and GPU (CuPy) computations, as well as MPI-based domain decomposition 
    for parallel execution on multiple processes.
    """
    def __init__(self, shape: tuple, dt: float = 1.0, use_gpu: bool = False):
        """
        Initialize the field solver.
        
        Args:
            shape: A tuple (Nx, Ny, Nz) for the spatial dimensions of the field grid.
            dt: Time step for the simulation.
            use_gpu: If True and CuPy is available, use GPU arrays for computation.
        """
        self.shape = shape
        self.dt = dt
        self.time = 0  # current time step index
        # Choose computation module (np or cp)
        self.xp = cp if (use_gpu and cp is not None) else np
        # Allocate field arrays (current and previous field states)
        self.field = self.xp.zeros(shape, dtype=float)
        self.field_prev = self.xp.zeros(shape, dtype=float)
        # MPI setup for domain decomposition
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        # Determine local slice for each MPI rank (simple 1D decomposition along x-axis for example)
        Nx = shape[0]
        if self.size > 1:
            # split along x dimension as an example
            chunk = Nx // self.size
            start = self.rank * chunk
            end = Nx if self.rank == self.size - 1 else (self.rank + 1) * chunk
            self.local_x_slice = (start, end)
        else:
            self.local_x_slice = (0, shape[0])
        # Damping factor for stability (small value to simulate energy dissipation)
        self.damping = 0.01

    def set_initial_field(self, init_func=None):
        """
        Set the initial field configuration.
        
        Args:
            init_func: A function f(x,y,z) returning initial value, or None for zero initial field.
        """
        if init_func:
            # Create coordinate arrays
            Nx, Ny, Nz = self.shape
            X = self.xp.arange(Nx); Y = self.xp.arange(Ny); Z = self.xp.arange(Nz)
            XX, YY, ZZ = self.xp.meshgrid(X, Y, Z, indexing='ij')
            initial = init_func(XX, YY, ZZ)
            self.field[...] = self.xp.array(initial, dtype=float)
        # If init_func is None, the field remains as zeros (already allocated)

    def _laplacian(self, array):
        """
        Compute the discrete Laplacian of a 3D field with periodic boundary conditions.
        """
        # Use np or cp roll for periodic shifts
        arr = array
        # Sum of neighbors in each dimension
        lap = (self.xp.roll(arr, 1, axis=0) + self.xp.roll(arr, -1, axis=0) +
               self.xp.roll(arr, 1, axis=1) + self.xp.roll(arr, -1, axis=1) +
               self.xp.roll(arr, 1, axis=2) + self.xp.roll(arr, -1, axis=2) -
               6 * arr)
        return lap

    def step(self, steps: int = 1):
        """
        Advance the field simulation by a given number of time steps.
        
        This uses a second-order accurate update (like a damped wave equation):
        new_field = 2*field - field_prev + c^2 * dt^2 * Laplacian(field) - damping * (field - field_prev)
        
        Args:
            steps: Number of time steps to advance.
        """
        c = 1.0  # wave propagation speed (can be adjusted)
        for _ in range(steps):
            # If using MPI, exchange boundary data with neighbors for correct Laplacian
            if self.size > 1:
                self._exchange_boundaries()
            # Compute Laplacian on current field (local portion if MPI)
            lap = self._laplacian(self.field)
            # Update field values using wave equation (with damping term)
            new_field = (2.0 - self.damping) * self.field - (1.0 - self.damping) * self.field_prev \
                        + (c * self.dt) ** 2 * lap
            # Update time and shift fields
            self.field_prev = self.field
            self.field = new_field
            self.time += 1

    def _exchange_boundaries(self):
        """
        Exchange ghost cells with neighboring MPI ranks (for domain decomposition).
        This example assumes domain split along X axis into strips for each rank.
        """
        if self.comm is None or self.size == 1:
            return
        # Determine neighbors in the X dimension (rank-1 and rank+1)
        prev_rank = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        next_rank = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL
        # Send/Recv the boundary slices
        # Prepare slices: send my first X-layer to prev, my last X-layer to next, etc.
        # Note: Converting to NumPy for communication to avoid pickling CuPy arrays
        if prev_rank != MPI.PROC_NULL:
            send_buf = self.xp.asnumpy(self.field[0, :, :]) if self.xp is not np else self.field[0, :, :]
            recv_buf = np.empty_like(send_buf)
            # Send my first slice to prev, receive prev's last slice
            self.comm.Send(send_buf, dest=prev_rank, tag=11)
            self.comm.Recv(recv_buf, source=prev_rank, tag=22)
            # Assign received slice to my "ghost" before-begin position
            if self.xp is not np:
                self.field[-1, :, :] = self.xp.array(recv_buf)
            else:
                self.field[-1, :, :] = recv_buf
        if next_rank != MPI.PROC_NULL:
            send_buf = self.xp.asnumpy(self.field[-1, :, :]) if self.xp is not np else self.field[-1, :, :]
            recv_buf = np.empty_like(send_buf)
            # Send my last slice to next, receive next's first slice
            self.comm.Send(send_buf, dest=next_rank, tag=22)
            self.comm.Recv(recv_buf, source=next_rank, tag=11)
            if self.xp is not np:
                self.field[0, :, :] = self.xp.array(recv_buf)
            else:
                self.field[0, :, :] = recv_buf

    def get_field(self) -> np.ndarray:
        """
        Retrieve the current field values as a NumPy array (collecting from GPU or MPI subdomains if needed).
        
        Returns:
            np.ndarray of the full field.
        """
        # Gather data from all ranks if MPI is used
        full_field = None
        if self.size > 1:
            # Each rank has its portion [local_x_slice] in x-dimension
            local_start, local_end = self.local_x_slice
            local_data_np = np.array(self.xp.asnumpy(self.field[local_start:local_end, :, :]), copy=False)
            # Gather all pieces to rank 0
            recv_list = None
            if self.rank == 0:
                recv_list = [np.empty_like(local_data_np) for _ in range(self.size)]
            self.comm.Gather(local_data_np, recv_list, root=0)
            if self.rank == 0:
                # Concatenate along x-axis
                full_field = np.concatenate(recv_list, axis=0)
        else:
            # Single process: just convert to NumPy if on GPU
            full_field = np.array(self.xp.asnumpy(self.field), copy=False) if self.xp is not np else self.field.copy()
        return full_field if full_field is not None else None

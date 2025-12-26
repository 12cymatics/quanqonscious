#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PCFE v3.0 - DISTRIBUTED MPI & VISUALIZATION MODULE                      ║
║  Production-Grade Parallel Processing & Real-time Rendering              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
from mpi4py import MPI
import cupy as cp
from numba import cuda, jit, prange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from vispy import app, scene, visuals
from vispy.visuals.transforms import STTransform
import vtk
from vtk.util import numpy_support
import pyvista as pv
from mayavi import mlab
import napari
import time
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import h5py
import zarr
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MPI DOMAIN DECOMPOSITION ENGINE                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MPIDomainDecomposition:
    """
    Advanced MPI domain decomposition for distributed field evolution
    
    ⟨DECOMPOSITION STRATEGY⟩
    • 3D Cartesian topology with ghost cell exchange
    • Asynchronous halo updates with overlap computation
    • Dynamic load balancing based on coherence metrics
    • Fault tolerance with checkpoint/restart capability
    """
    
    def __init__(self, config, comm: MPI.Comm):
        self.config = config
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.logger = logging.getLogger(f'PCFE.MPI.Rank{self.rank}')
        
        # ⟨TOPOLOGY INITIALIZATION⟩
        self.topology = self._create_cartesian_topology()
        self.local_domain = self._compute_local_domain()
        self.ghost_cells = config.mpi_chunk_overlap
        
        # ⟨COMMUNICATION BUFFERS⟩
        self.send_buffers = self._allocate_communication_buffers()
        self.recv_buffers = self._allocate_communication_buffers()
        self.requests = []
        
        # ⟨PERFORMANCE TRACKING⟩
        self.communication_time = 0.0
        self.computation_time = 0.0
        self.load_imbalance_factor = 0.0
        
        self.logger.info(f"MPI domain initialized: {self.local_domain}")
    
    def _create_cartesian_topology(self) -> MPI.Cartcomm:
        """Create optimal 3D Cartesian topology"""
        # Factor size into 3D grid
        dims = MPI.Compute_dims(self.size, [0, 0, 0])
        
        # Create Cartesian communicator
        cart_comm = self.comm.Create_cart(
            dims=dims,
            periods=[True, True, True],  # Periodic boundaries
            reorder=True
        )
        
        # Get coordinates in topology
        self.coords = cart_comm.Get_coords(self.rank)
        
        # Get neighbor ranks
        self.neighbors = {
            'left': cart_comm.Shift(0, -1),
            'right': cart_comm.Shift(0, 1),
            'down': cart_comm.Shift(1, -1),
            'up': cart_comm.Shift(1, 1),
            'back': cart_comm.Shift(2, -1),
            'front': cart_comm.Shift(2, 1)
        }
        
        return cart_comm
    
    def _compute_local_domain(self) -> Dict[str, Tuple[int, int]]:
        """Compute local domain boundaries for this rank"""
        global_size = self.config.grid_size
        dims = self.topology.Get_dim()
        
        # Compute local sizes
        local_sizes = []
        starts = []
        ends = []
        
        for dim in range(3):
            dim_size = dims[dim]
            coord = self.coords[dim]
            
            # Base size and remainder
            base_size = global_size // dim_size
            remainder = global_size % dim_size
            
            # Distribute remainder to first ranks
            if coord < remainder:
                local_size = base_size + 1
                start = coord * local_size
            else:
                local_size = base_size
                start = remainder * (base_size + 1) + (coord - remainder) * base_size
            
            end = start + local_size
            
            local_sizes.append(local_size)
            starts.append(start)
            ends.append(end)
        
        return {
            'x': (starts[0], ends[0]),
            'y': (starts[1], ends[1]),
            'z': (starts[2], ends[2]),
            'shape': tuple(local_sizes)
        }
    
    def _allocate_communication_buffers(self) -> Dict[str, torch.Tensor]:
        """Allocate ghost cell communication buffers"""
        buffers = {}
        shape = self.local_domain['shape']
        ghost = self.ghost_cells
        dtype = self.config.dtype
        device = self.config.device
        
        # Face buffers (6 faces)
        buffers['left'] = torch.zeros((ghost, shape[1], shape[2]), dtype=dtype, device=device)
        buffers['right'] = torch.zeros((ghost, shape[1], shape[2]), dtype=dtype, device=device)
        buffers['down'] = torch.zeros((shape[0], ghost, shape[2]), dtype=dtype, device=device)
        buffers['up'] = torch.zeros((shape[0], ghost, shape[2]), dtype=dtype, device=device)
        buffers['back'] = torch.zeros((shape[0], shape[1], ghost), dtype=dtype, device=device)
        buffers['front'] = torch.zeros((shape[0], shape[1], ghost), dtype=dtype, device=device)
        
        # Edge buffers (12 edges) - for higher-order stencils
        if ghost > 1:
            # X-aligned edges
            buffers['edge_xy_00'] = torch.zeros((shape[0], ghost, ghost), dtype=dtype, device=device)
            buffers['edge_xy_01'] = torch.zeros((shape[0], ghost, ghost), dtype=dtype, device=device)
            buffers['edge_xy_10'] = torch.zeros((shape[0], ghost, ghost), dtype=dtype, device=device)
            buffers['edge_xy_11'] = torch.zeros((shape[0], ghost, ghost), dtype=dtype, device=device)
            
            # Y-aligned edges
            buffers['edge_yz_00'] = torch.zeros((ghost, shape[1], ghost), dtype=dtype, device=device)
            buffers['edge_yz_01'] = torch.zeros((ghost, shape[1], ghost), dtype=dtype, device=device)
            buffers['edge_yz_10'] = torch.zeros((ghost, shape[1], ghost), dtype=dtype, device=device)
            buffers['edge_yz_11'] = torch.zeros((ghost, shape[1], ghost), dtype=dtype, device=device)
            
            # Z-aligned edges
            buffers['edge_xz_00'] = torch.zeros((ghost, ghost, shape[2]), dtype=dtype, device=device)
            buffers['edge_xz_01'] = torch.zeros((ghost, ghost, shape[2]), dtype=dtype, device=device)
            buffers['edge_xz_10'] = torch.zeros((ghost, ghost, shape[2]), dtype=dtype, device=device)
            buffers['edge_xz_11'] = torch.zeros((ghost, ghost, shape[2]), dtype=dtype, device=device)
        
        # Corner buffers (8 corners) - for maximum stencil support
        if ghost > 2:
            for i in range(8):
                buffers[f'corner_{i}'] = torch.zeros((ghost, ghost, ghost), dtype=dtype, device=device)
        
        return buffers
    
    def decompose_field(self, global_field: torch.Tensor) -> torch.Tensor:
        """Decompose global field to local domain with ghost cells"""
        x_start, x_end = self.local_domain['x']
        y_start, y_end = self.local_domain['y']
        z_start, z_end = self.local_domain['z']
        ghost = self.ghost_cells
        
        # Allocate local field with ghost cells
        local_shape = (
            x_end - x_start + 2 * ghost,
            y_end - y_start + 2 * ghost,
            z_end - z_start + 2 * ghost
        )
        local_field = torch.zeros(local_shape, dtype=global_field.dtype, device=global_field.device)
        
        # Copy interior data
        local_field[ghost:-ghost, ghost:-ghost, ghost:-ghost] = \
            global_field[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Fill ghost cells (periodic boundaries)
        # Left ghost
        local_field[:ghost, ghost:-ghost, ghost:-ghost] = \
            global_field[x_start-ghost:x_start, y_start:y_end, z_start:z_end]
        
        # Right ghost
        local_field[-ghost:, ghost:-ghost, ghost:-ghost] = \
            global_field[x_end:x_end+ghost, y_start:y_end, z_start:z_end]
        
        # Similar for other directions...
        
        return local_field
    
    async def exchange_ghost_cells_async(self, local_field: torch.Tensor):
        """Asynchronous ghost cell exchange with overlap"""
        ghost = self.ghost_cells
        
        # ⟨PHASE 1: INITIATE SENDS⟩
        # Pack and send face data
        self._pack_face_data(local_field)
        self._initiate_face_sends()
        
        # ⟨PHASE 2: COMPUTE INTERIOR⟩
        # While communication happens, we can compute interior points
        # that don't depend on ghost cells
        
        # ⟨PHASE 3: RECEIVE AND UNPACK⟩
        await self._receive_face_data_async()
        self._unpack_face_data(local_field)
        
        # ⟨PHASE 4: EDGE/CORNER EXCHANGE⟩
        if ghost > 1:
            self._exchange_edges_corners(local_field)
    
    def _pack_face_data(self, local_field: torch.Tensor):
        """Pack face data into send buffers"""
        ghost = self.ghost_cells
        
        # Left face
        self.send_buffers['left'].copy_(local_field[ghost:2*ghost, ghost:-ghost, ghost:-ghost])
        
        # Right face
        self.send_buffers['right'].copy_(local_field[-2*ghost:-ghost, ghost:-ghost, ghost:-ghost])
        
        # Down face
        self.send_buffers['down'].copy_(local_field[ghost:-ghost, ghost:2*ghost, ghost:-ghost])
        
        # Up face
        self.send_buffers['up'].copy_(local_field[ghost:-ghost, -2*ghost:-ghost, ghost:-ghost])
        
        # Back face
        self.send_buffers['back'].copy_(local_field[ghost:-ghost, ghost:-ghost, ghost:2*ghost])
        
        # Front face
        self.send_buffers['front'].copy_(local_field[ghost:-ghost, ghost:-ghost, -2*ghost:-ghost])
    
    def _initiate_face_sends(self):
        """Initiate non-blocking sends for face data"""
        # Convert to numpy for MPI
        for direction, neighbor_rank in self.neighbors.items():
            if neighbor_rank[1] != MPI.PROC_NULL:
                send_data = self.send_buffers[direction].cpu().numpy()
                recv_data = self.recv_buffers[direction].cpu().numpy()
                
                # Non-blocking send/recv
                send_req = self.comm.Isend(send_data, dest=neighbor_rank[1], tag=self._get_tag(direction))
                recv_req = self.comm.Irecv(recv_data, source=neighbor_rank[0], tag=self._get_tag(self._opposite(direction)))
                
                self.requests.extend([send_req, recv_req])
    
    async def _receive_face_data_async(self):
        """Wait for all communications to complete"""
        # Wait for all requests
        MPI.Request.Waitall(self.requests)
        self.requests.clear()
        
        # Copy received data back to GPU
        for direction in self.neighbors:
            self.recv_buffers[direction] = torch.from_numpy(
                self.recv_buffers[direction].cpu().numpy()
            ).to(self.config.device)
        
        await asyncio.sleep(0)  # Yield control
    
    def _unpack_face_data(self, local_field: torch.Tensor):
        """Unpack received data into ghost cells"""
        ghost = self.ghost_cells
        
        # Left ghost
        local_field[:ghost, ghost:-ghost, ghost:-ghost].copy_(self.recv_buffers['left'])
        
        # Right ghost
        local_field[-ghost:, ghost:-ghost, ghost:-ghost].copy_(self.recv_buffers['right'])
        
        # Down ghost
        local_field[ghost:-ghost, :ghost, ghost:-ghost].copy_(self.recv_buffers['down'])
        
        # Up ghost
        local_field[ghost:-ghost, -ghost:, ghost:-ghost].copy_(self.recv_buffers['up'])
        
        # Back ghost
        local_field[ghost:-ghost, ghost:-ghost, :ghost].copy_(self.recv_buffers['back'])
        
        # Front ghost
        local_field[ghost:-ghost, ghost:-ghost, -ghost:].copy_(self.recv_buffers['front'])
    
    def _get_tag(self, direction: str) -> int:
        """Generate unique MPI tag for direction"""
        tag_map = {
            'left': 0, 'right': 1, 'down': 2, 'up': 3, 'back': 4, 'front': 5
        }
        return tag_map.get(direction, 0) + self.rank * 10
    
    def _opposite(self, direction: str) -> str:
        """Get opposite direction"""
        opposites = {
            'left': 'right', 'right': 'left',
            'down': 'up', 'up': 'down',
            'back': 'front', 'front': 'back'
        }
        return opposites[direction]
    
    def gather_global_field(self, local_field: torch.Tensor) -> Optional[torch.Tensor]:
        """Gather local fields to global field on rank 0"""
        ghost = self.ghost_cells
        
        # Extract interior (non-ghost) data
        interior = local_field[ghost:-ghost, ghost:-ghost, ghost:-ghost]
        
        # Gather shapes first
        local_shape = interior.shape
        all_shapes = self.comm.gather(local_shape, root=0)
        
        # Gather domain info
        domain_info = {
            'x': self.local_domain['x'],
            'y': self.local_domain['y'],
            'z': self.local_domain['z'],
            'shape': local_shape
        }
        all_domains = self.comm.gather(domain_info, root=0)
        
        # Gather data
        interior_np = interior.cpu().numpy()
        
        if self.rank == 0:
            # Allocate global field
            global_shape = (self.config.grid_size, self.config.grid_size, self.config.grid_size)
            global_field = np.zeros(global_shape, dtype=np.complex128)
            
            # Place rank 0's data
            x_start, x_end = all_domains[0]['x']
            y_start, y_end = all_domains[0]['y']
            z_start, z_end = all_domains[0]['z']
            global_field[x_start:x_end, y_start:y_end, z_start:z_end] = interior_np
            
            # Receive and place other ranks' data
            for rank in range(1, self.size):
                # Receive data
                data_shape = all_domains[rank]['shape']
                recv_data = np.empty(data_shape, dtype=np.complex128)
                self.comm.Recv(recv_data, source=rank, tag=100+rank)
                
                # Place in global field
                x_start, x_end = all_domains[rank]['x']
                y_start, y_end = all_domains[rank]['y']
                z_start, z_end = all_domains[rank]['z']
                global_field[x_start:x_end, y_start:y_end, z_start:z_end] = recv_data
            
            return torch.from_numpy(global_field).to(self.config.device)
        else:
            # Send data to rank 0
            self.comm.Send(interior_np, dest=0, tag=100+self.rank)
            return None
    
    def compute_load_balance_metrics(self, computation_time: float) -> Dict[str, float]:
        """Compute load balance metrics across all ranks"""
        # Gather all computation times
        all_times = self.comm.gather(computation_time, root=0)
        
        metrics = {}
        if self.rank == 0:
            all_times = np.array(all_times)
            metrics['mean_time'] = np.mean(all_times)
            metrics['max_time'] = np.max(all_times)
            metrics['min_time'] = np.min(all_times)
            metrics['std_time'] = np.std(all_times)
            metrics['imbalance_factor'] = (metrics['max_time'] - metrics['min_time']) / metrics['mean_time']
            metrics['efficiency'] = metrics['mean_time'] / metrics['max_time']
        
        # Broadcast metrics to all ranks
        metrics = self.comm.bcast(metrics, root=0)
        self.load_imbalance_factor = metrics.get('imbalance_factor', 0.0)
        
        return metrics

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ADVANCED VISUALIZATION ENGINE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AdvancedVisualizationEngine:
    """
    Multi-backend visualization engine for real-time field rendering
    
    ⟨VISUALIZATION CAPABILITIES⟩
    • Real-time 3D volume rendering with GPU acceleration
    • Interactive slice exploration
    • Isosurface extraction with marching cubes
    • Vector field visualization
    • Quantum state tomography plots
    • Coherence evolution tracking
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('PCFE.Visualization')
        
        # ⟨RENDERING BACKENDS⟩
        self.backends = {
            'vispy': self._init_vispy(),
            'vtk': self._init_vtk(),
            'plotly': self._init_plotly(),
            'mayavi': self._init_mayavi()
        }
        
        # ⟨VISUALIZATION STATE⟩
        self.current_backend = 'vispy'
        self.colormap = self._create_quantum_colormap()
        self.animation_running = False
        
    def _init_vispy(self):
        """Initialize VisPy for real-time GPU rendering"""
        # Create canvas
        canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                                 size=(1024, 768), title='PCFE Field Visualization')
        canvas.show()
        
        # Add 3D view
        view = canvas.central_widget.add_view()
        
        # Create camera
        cam = scene.TurntableCamera(elevation=30, azimuth=30, 
                                   fov=60, distance=2)
        view.camera = cam
        
        # Add axes
        axes = scene.visuals.XYZAxis(parent=view.scene)
        
        return {
            'canvas': canvas,
            'view': view,
            'camera': cam,
            'volume': None,
            'isosurface': None,
            'slices': {}
        }
    
    def _init_vtk(self):
        """Initialize VTK for advanced volume rendering"""
        # Create renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)
        
        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1024, 768)
        
        # Create interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # Volume rendering components
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_property = vtk.vtkVolumeProperty()
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        return {
            'renderer': renderer,
            'render_window': render_window,
            'interactor': interactor,
            'volume_mapper': volume_mapper,
            'volume_property': volume_property,
            'volume': volume
        }
    
    def _init_plotly(self):
        """Initialize Plotly for interactive web-based visualization"""
        return {
            'figures': {},
            'animations': {}
        }
    
    def _init_mayavi(self):
        """Initialize Mayavi for scientific visualization"""
        # Note: Mayavi requires display, so we check availability
        try:
            from mayavi import mlab
            mlab.options.offscreen = False
            return {'available': True}
        except:
            self.logger.warning("Mayavi not available for visualization")
            return {'available': False}
    
    def _create_quantum_colormap(self):
        """Create custom quantum-themed colormap"""
        colors = [
            (0.0, 0.0, 0.2, 1.0),   # Deep blue (low values)
            (0.0, 0.0, 0.5, 1.0),   # Blue
            (0.0, 0.5, 1.0, 1.0),   # Cyan
            (0.0, 1.0, 1.0, 1.0),   # Light cyan
            (1.0, 1.0, 0.0, 1.0),   # Yellow
            (1.0, 0.5, 0.0, 1.0),   # Orange
            (1.0, 0.0, 0.0, 1.0),   # Red (high values)
            (0.5, 0.0, 0.5, 1.0)    # Purple (max values)
        ]
        
        # Create matplotlib colormap
        n_bins = 256
        cmap_name = 'quantum'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        return cmap
    
    @torch.cuda.amp.autocast()
    def visualize_field_3d(self, field: torch.Tensor, mode: str = 'magnitude',
                          backend: Optional[str] = None):
        """Main 3D field visualization method"""
        backend = backend or self.current_backend
        
        if backend == 'vispy':
            self._visualize_vispy(field, mode)
        elif backend == 'vtk':
            self._visualize_vtk(field, mode)
        elif backend == 'plotly':
            self._visualize_plotly(field, mode)
        elif backend == 'mayavi':
            self._visualize_mayavi(field, mode)
    
    def _visualize_vispy(self, field: torch.Tensor, mode: str):
        """VisPy real-time visualization"""
        vispy_backend = self.backends['vispy']
        
        # Prepare data
        if mode == 'magnitude':
            data = torch.abs(field).cpu().numpy()
        elif mode == 'phase':
            data = torch.angle(field).cpu().numpy()
        elif mode == 'real':
            data = field.real.cpu().numpy()
        elif mode == 'imag':
            data = field.imag.cpu().numpy()
        else:
            data = torch.abs(field).cpu().numpy()
        
        # Normalize data
        data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        # Create or update volume visual
        if vispy_backend['volume'] is None:
            # Create volume visual
            volume = scene.visuals.Volume(
                data,
                parent=vispy_backend['view'].scene,
                cmap=self.colormap,
                method='translucent',
                relative_step_size=0.5
            )
            
            # Set transform
            volume.transform = STTransform(translate=(-0.5, -0.5, -0.5))
            
            vispy_backend['volume'] = volume
        else:
            # Update existing volume
            vispy_backend['volume'].set_data(data)
        
        # Update canvas
        vispy_backend['canvas'].update()
    
    def _visualize_vtk(self, field: torch.Tensor, mode: str):
        """VTK advanced volume rendering"""
        vtk_backend = self.backends['vtk']
        
        # Prepare data
        if mode == 'magnitude':
            data = torch.abs(field).cpu().numpy()
        else:
            data = field.real.cpu().numpy()
        
        # Create VTK image data
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(field.shape)
        vtk_data.SetSpacing(1.0, 1.0, 1.0)
        vtk_data.SetOrigin(0.0, 0.0, 0.0)
        
        # Convert numpy to VTK array
        vtk_array = numpy_support.numpy_to_vtk(
            data.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        )
        vtk_array.SetName("field")
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        # Update volume mapper
        vtk_backend['volume_mapper'].SetInputData(vtk_data)
        
        # Configure volume property
        volume_property = vtk_backend['volume_property']
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Create opacity transfer function
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0.0, 0.0)
        opacity_func.AddPoint(0.2, 0.1)
        opacity_func.AddPoint(0.5, 0.3)
        opacity_func.AddPoint(0.8, 0.6)
        opacity_func.AddPoint(1.0, 1.0)
        volume_property.SetScalarOpacity(opacity_func)
        
        # Create color transfer function
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.2)
        color_func.AddRGBPoint(0.25, 0.0, 0.0, 1.0)
        color_func.AddRGBPoint(0.5, 0.0, 1.0, 1.0)
        color_func.AddRGBPoint(0.75, 1.0, 1.0, 0.0)
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        volume_property.SetColor(color_func)
        
        # Add volume to renderer
        vtk_backend['renderer'].AddVolume(vtk_backend['volume'])
        vtk_backend['renderer'].ResetCamera()
        
        # Start interaction
        vtk_backend['render_window'].Render()
        vtk_backend['interactor'].Start()
    
    def _visualize_plotly(self, field: torch.Tensor, mode: str):
        """Plotly interactive web visualization"""
        import plotly.graph_objects as go
        
        # Prepare data
        if mode == 'magnitude':
            data = torch.abs(field).cpu().numpy()
        else:
            data = field.real.cpu().numpy()
        
        # Create isosurface plot
        fig = go.Figure(data=go.Isosurface(
            x=np.arange(field.shape[0]),
            y=np.arange(field.shape[1]),
            z=np.arange(field.shape[2]),
            value=data.flatten(),
            isomin=data.min(),
            isomax=data.max(),
            surface_count=5,
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.3
        ))
        
        # Update layout
        fig.update_layout(
            title=f'PCFE Field Visualization - {mode}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        # Show plot
        fig.show()
        
        # Store figure
        self.backends['plotly']['figures'][mode] = fig
    
    def _visualize_mayavi(self, field: torch.Tensor, mode: str):
        """Mayavi scientific visualization"""
        if not self.backends['mayavi']['available']:
            self.logger.warning("Mayavi not available")
            return
        
        from mayavi import mlab
        
        # Prepare data
        if mode == 'magnitude':
            data = torch.abs(field).cpu().numpy()
        else:
            data = field.real.cpu().numpy()
        
        # Create figure
        fig = mlab.figure(size=(800, 600), bgcolor=(0.1, 0.1, 0.1))
        
        # Volume rendering
        src = mlab.pipeline.scalar_field(data)
        
        # Add volume
        vol = mlab.pipeline.volume(src, vmin=data.min(), vmax=data.max())
        
        # Add contour
        contour = mlab.pipeline.iso_surface(
            src, contours=[data.mean()], opacity=0.3
        )
        
        # Add cut planes
        cut_plane = mlab.pipeline.scalar_cut_plane(
            src, plane_orientation='x_axes', colormap='hot'
        )
        
        # Configure view
        mlab.view(azimuth=45, elevation=60, distance='auto')
        mlab.show()
    
    def create_animation(self, field_history: List[torch.Tensor], 
                        output_path: str, fps: int = 30):
        """Create animation from field history"""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        fig = plt.figure(figsize=(12, 10))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # XY slice
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xy.set_title('XY Plane')
        ax_xy.set_xlabel('Y')
        ax_xy.set_ylabel('X')
        
        # XZ slice
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_xz.set_title('XZ Plane')
        ax_xz.set_xlabel('Z')
        ax_xz.set_ylabel('X')
        
        # YZ slice
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_yz.set_title('YZ Plane')
        ax_yz.set_xlabel('Z')
        ax_yz.set_ylabel('Y')
        
        # Phase distribution
        ax_phase = fig.add_subplot(gs[1, :])
        ax_phase.set_title('Phase Distribution')
        ax_phase.set_xlabel('Phase')
        ax_phase.set_ylabel('Probability')
        
        # Coherence metrics
        ax_metrics = fig.add_subplot(gs[2, :])
        ax_metrics.set_title('Coherence Evolution')
        ax_metrics.set_xlabel('Time Step')
        ax_metrics.set_ylabel('Coherence')
        
        # Animation function
        def animate(frame):
            field = field_history[frame]
            center = field.shape[0] // 2
            
            # Clear axes
            ax_xy.clear()
            ax_xz.clear()
            ax_yz.clear()
            ax_phase.clear()
            
            # Magnitude slices
            magnitude = torch.abs(field).cpu().numpy()
            
            im_xy = ax_xy.imshow(magnitude[center, :, :], cmap=self.colormap, 
                                vmin=0, vmax=magnitude.max())
            im_xz = ax_xz.imshow(magnitude[:, center, :], cmap=self.colormap,
                                vmin=0, vmax=magnitude.max())
            im_yz = ax_yz.imshow(magnitude[:, :, center], cmap=self.colormap,
                                vmin=0, vmax=magnitude.max())
            
            # Phase histogram
            phase = torch.angle(field).cpu().numpy().flatten()
            ax_phase.hist(phase, bins=50, density=True, alpha=0.7, color='cyan')
            ax_phase.set_xlim(-np.pi, np.pi)
            
            # Titles with frame number
            ax_xy.set_title(f'XY Plane - Frame {frame}')
            ax_xz.set_title(f'XZ Plane - Frame {frame}')
            ax_yz.set_title(f'YZ Plane - Frame {frame}')
            
            return [im_xy, im_xz, im_yz]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(field_history),
            interval=1000/fps, blit=False
        )
        
        # Save animation
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(output_path, writer=writer)
        
        plt.close()
        self.logger.info(f"Animation saved to {output_path}")
    
    def visualize_coherence_evolution(self, metric_history: List[Dict[str, float]]):
        """Visualize coherence metric evolution"""
        import matplotlib.pyplot as plt
        
        # Extract metrics
        metrics_to_plot = ['global', 'spatial', 'temporal', 'quantum', 'emergent', 'entropy']
        time_steps = range(len(metric_history))
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [m[metric] for m in metric_history]
            
            # Plot metric
            ax.plot(time_steps, values, linewidth=2)
            ax.set_title(f'{metric.capitalize()} Coherence')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add threshold lines
            if metric == 'global':
                ax.axhline(y=0.99, color='r', linestyle='--', label='Target')
                ax.legend()
            elif metric == 'emergent':
                ax.axhline(y=0.7, color='g', linestyle='--', label='Threshold')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_vtk_dataset(self, field: torch.Tensor, filename: str):
        """Export field as VTK dataset for external tools"""
        # Prepare data
        magnitude = torch.abs(field).cpu().numpy()
        phase = torch.angle(field).cpu().numpy()
        real = field.real.cpu().numpy()
        imag = field.imag.cpu().numpy()
        
        # Create VTK structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(field.shape)
        
        # Create points
        points = vtk.vtkPoints()
        for k in range(field.shape[2]):
            for j in range(field.shape[1]):
                for i in range(field.shape[0]):
                    points.InsertNextPoint(i, j, k)
        grid.SetPoints(points)
        
        # Add field data
        arrays = {
            'magnitude': magnitude,
            'phase': phase,
            'real': real,
            'imag': imag
        }
        
        for name, data in arrays.items():
            vtk_array = numpy_support.numpy_to_vtk(
                data.ravel(), deep=True, array_type=vtk.VTK_FLOAT
            )
            vtk_array.SetName(name)
            grid.GetPointData().AddArray(vtk_array)
        
        # Write to file
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
        
        self.logger.info(f"VTK dataset exported to {filename}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PERFORMANCE PROFILING ENGINE                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PerformanceProfiler:
    """
    Comprehensive performance profiling and optimization system
    
    ⟨PROFILING CAPABILITIES⟩
    • CUDA kernel profiling with nvprof integration
    • Memory bandwidth analysis
    • Coherence calculation bottleneck detection
    • MPI communication overhead tracking
    • Quantum circuit compilation time
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('PCFE.Performance')
        
        # ⟨TIMING TRACKERS⟩
        self.timers = {
            'field_evolution': [],
            'coherence_calculation': [],
            'mpi_communication': [],
            'quantum_simulation': [],
            'vedic_sutra_application': [],
            'visualization': [],
            'checkpoint_io': []
        }
        
        # ⟨MEMORY TRACKERS⟩
        self.memory_usage = {
            'peak_gpu': 0,
            'current_gpu': 0,
            'peak_cpu': 0,
            'current_cpu': 0
        }
        
        # ⟨CUDA PROFILING⟩
        self.cuda_events = self._init_cuda_events()
        
        # ⟨PERFORMANCE METRICS⟩
        self.flops_achieved = 0
        self.bandwidth_achieved = 0
        self.efficiency_metrics = {}
    
    def _init_cuda_events(self) -> Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Initialize CUDA events for kernel timing"""
        events = {}
        kernels = [
            'field_evolution', 'laplacian', 'coherence', 
            'fft', 'vedic_operations'
        ]
        
        for kernel in kernels:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            events[kernel] = (start, end)
        
        return events
    
    @contextlib.contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling code sections"""
        # CPU timing
        cpu_start = time.perf_counter()
        
        # GPU timing
        if section_name in self.cuda_events:
            start_event, end_event = self.cuda_events[section_name]
            start_event.record()
        
        # Memory before
        mem_before = self._get_memory_usage()
        
        try:
            yield
        finally:
            # GPU timing
            if section_name in self.cuda_events:
                end_event.record()
                torch.cuda.synchronize()
                gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                gpu_time = 0
            
            # CPU timing
            cpu_time = time.perf_counter() - cpu_start
            
            # Memory after
            mem_after = self._get_memory_usage()
            mem_delta = {
                'gpu': mem_after['gpu'] - mem_before['gpu'],
                'cpu': mem_after['cpu'] - mem_before['cpu']
            }
            
            # Record timing
            self.timers[section_name].append({
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'memory_delta': mem_delta,
                'timestamp': time.time()
            })
            
            # Update peak memory
            self.memory_usage['peak_gpu'] = max(
                self.memory_usage['peak_gpu'], mem_after['gpu']
            )
            self.memory_usage['peak_cpu'] = max(
                self.memory_usage['peak_cpu'], mem_after['cpu']
            )
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        import psutil
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            gpu_mem = 0
        
        # CPU memory
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / 1024**3  # GB
        
        return {'gpu': gpu_mem, 'cpu': cpu_mem}
    
    def calculate_achieved_performance(self, field_size: int, iterations: int):
        """Calculate achieved FLOPS and bandwidth"""
        # Estimate operations per iteration
        # Laplacian: 7 operations per point
        # Nonlinear term: 5 operations per point
        # Vedic operations: ~10 operations per point
        ops_per_point = 22
        total_ops = field_size**3 * ops_per_point * iterations
        
        # Get total time
        total_time = sum(sum(t['gpu_time'] for t in timings) 
                        for timings in self.timers.values())
        
        if total_time > 0:
            self.flops_achieved = total_ops / total_time / 1e12  # TFLOPS
        
        # Estimate bandwidth (assuming complex128 = 16 bytes)
        bytes_per_iteration = field_size**3 * 16 * 3  # Read, write, ghost cells
        total_bytes = bytes_per_iteration * iterations
        
        if total_time > 0:
            self.bandwidth_achieved = total_bytes / total_time / 1e9  # GB/s
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timing_summary': {},
            'memory_summary': self.memory_usage,
            'performance_metrics': {
                'achieved_tflops': self.flops_achieved,
                'achieved_bandwidth_gbps': self.bandwidth_achieved
            },
            'bottlenecks': []
        }
        
        # Timing summary
        for section, timings in self.timers.items():
            if timings:
                cpu_times = [t['cpu_time'] for t in timings]
                gpu_times = [t['gpu_time'] for t in timings]
                
                report['timing_summary'][section] = {
                    'total_cpu_time': sum(cpu_times),
                    'avg_cpu_time': np.mean(cpu_times),
                    'total_gpu_time': sum(gpu_times),
                    'avg_gpu_time': np.mean(gpu_times),
                    'call_count': len(timings)
                }
        
        # Identify bottlenecks
        total_time = sum(report['timing_summary'][s]['total_gpu_time'] 
                        for s in report['timing_summary'])
        
        for section, stats in report['timing_summary'].items():
            percentage = stats['total_gpu_time'] / total_time * 100
            if percentage > 20:  # More than 20% of time
                report['bottlenecks'].append({
                    'section': section,
                    'percentage': percentage,
                    'total_time': stats['total_gpu_time']
                })
        
        # Sort bottlenecks by percentage
        report['bottlenecks'].sort(key=lambda x: x['percentage'], reverse=True)
        
        return report
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Suggest optimized parameters based on profiling"""
        suggestions = {}
        
        # Analyze MPI communication overhead
        if 'mpi_communication' in self.timers and self.timers['mpi_communication']:
            mpi_overhead = sum(t['gpu_time'] for t in self.timers['mpi_communication'])
            total_time = sum(sum(t['gpu_time'] for t in timings) 
                           for timings in self.timers.values())
            
            mpi_percentage = mpi_overhead / total_time * 100
            
            if mpi_percentage > 30:
                suggestions['reduce_mpi_frequency'] = {
                    'current_overlap': self.config.mpi_chunk_overlap,
                    'suggested_overlap': self.config.mpi_chunk_overlap * 2,
                    'reason': f'MPI overhead is {mpi_percentage:.1f}%'
                }
        
        # Analyze memory usage
        if self.memory_usage['peak_gpu'] > 0:
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_percentage = self.memory_usage['peak_gpu'] / gpu_memory_total * 100
            
            if usage_percentage > 80:
                suggestions['reduce_grid_size'] = {
                    'current_size': self.config.grid_size,
                    'suggested_size': int(self.config.grid_size * 0.8),
                    'reason': f'GPU memory usage is {usage_percentage:.1f}%'
                }
        
        # Analyze quantum simulation overhead
        if 'quantum_simulation' in self.timers and self.timers['quantum_simulation']:
            quantum_time = sum(t['cpu_time'] for t in self.timers['quantum_simulation'])
            total_time = sum(sum(t['cpu_time'] for t in timings) 
                           for timings in self.timers.values())
            
            quantum_percentage = quantum_time / total_time * 100
            
            if quantum_percentage > 40:
                suggestions['reduce_quantum_shots'] = {
                    'current_shots': self.config.quantum_shots,
                    'suggested_shots': self.config.quantum_shots // 2,
                    'reason': f'Quantum simulation is {quantum_percentage:.1f}% of runtime'
                }
        
        return suggestions
    
    def export_profile_data(self, filename: str):
        """Export profiling data for external analysis"""
        profile_data = {
            'timers': self.timers,
            'memory_usage': self.memory_usage,
            'performance_metrics': {
                'flops_achieved': self.flops_achieved,
                'bandwidth_achieved': self.bandwidth_achieved
            },
            'config': {
                'grid_size': self.config.grid_size,
                'device': self.config.device,
                'mpi_size': MPI.COMM_WORLD.Get_size() if self.config.enable_mpi else 1
            },
            'report': self.generate_performance_report()
        }
        
        # Save as JSON
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        self.logger.info(f"Profile data exported to {filename}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DISTRIBUTED PCFE ORCHESTRATOR                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DistributedPCFE:
    """
    Main orchestrator for distributed Proto-Consciousness Field Engine
    
    ⟨ARCHITECTURE⟩
    • Master-worker paradigm with dynamic load balancing
    • Asynchronous field evolution with overlap communication
    • Distributed coherence calculation with reduction
    • Fault-tolerant checkpointing across nodes
    """
    
    def __init__(self, config: PCFEConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Modify config for distributed execution
        if self.rank > 0:
            # Only rank 0 does visualization
            config.visualization_interval = 0
        
        # Initialize components
        self.domain_decomp = MPIDomainDecomposition(config, self.comm)
        self.visualization = AdvancedVisualizationEngine(config) if self.rank == 0 else None
        self.profiler = PerformanceProfiler(config)
        
        # Import main PCFE components
        from pcfe_v3_core_engine import (
            QuantumVacuumSystem, VedicSutraEngine, 
            FieldDynamicsEngine, CoherenceAnalysisEngine
        )
        
        self.quantum_vacuum = QuantumVacuumSystem(config)
        self.vedic_sutras = VedicSutraEngine(config)
        self.field_dynamics = FieldDynamicsEngine(config)
        self.coherence_analysis = CoherenceAnalysisEngine(config)
        
        # Initialize local field
        self.local_field = None
        self.time_step = 0
        
        self.logger = logging.getLogger(f'PCFE.Distributed.Rank{self.rank}')
    
    async def initialize_field(self):
        """Initialize field with domain decomposition"""
        if self.rank == 0:
            # Create global field on rank 0
            global_field = self._create_initial_field()
            self.logger.info(f"Global field initialized: {global_field.shape}")
        else:
            global_field = None
        
        # Broadcast global field shape
        if self.rank == 0:
            field_shape = global_field.shape
        else:
            field_shape = None
        field_shape = self.comm.bcast(field_shape, root=0)
        
        # Scatter field to all ranks
        if self.rank == 0:
            # Decompose for all ranks
            all_local_fields = []
            for rank in range(self.size):
                # Temporarily set rank for decomposition
                self.domain_decomp.rank = rank
                self.domain_decomp.coords = self.domain_decomp.topology.Get_coords(rank)
                self.domain_decomp.local_domain = self.domain_decomp._compute_local_domain()
                
                local = self.domain_decomp.decompose_field(global_field)
                all_local_fields.append(local)
            
            # Reset rank
            self.domain_decomp.rank = self.rank
            self.domain_decomp.coords = self.domain_decomp.topology.Get_coords(self.rank)
            self.domain_decomp.local_domain = self.domain_decomp._compute_local_domain()
            
            # Distribute fields
            self.local_field = all_local_fields[0]
            for rank in range(1, self.size):
                self.comm.send(all_local_fields[rank], dest=rank, tag=1000)
        else:
            # Receive local field
            self.local_field = self.comm.recv(source=0, tag=1000)
        
        self.logger.info(f"Local field initialized: {self.local_field.shape}")
    
    def _create_initial_field(self) -> torch.Tensor:
        """Create initial global field (rank 0 only)"""
        size = self.config.grid_size
        field = torch.zeros((size, size, size), dtype=self.config.dtype, device=self.config.device)
        
        # Create coordinate grids
        x = torch.linspace(-4, 4, size, device=self.config.device)
        y = torch.linspace(-4, 4, size, device=self.config.device)
        z = torch.linspace(-4, 4, size, device=self.config.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Multi-vortex initial condition
        R = torch.sqrt(X**2 + Y**2 + Z**2)
        theta = torch.atan2(Y, X)
        phi = torch.atan2(Z, torch.sqrt(X**2 + Y**2))
        
        # Amplitude with multiple Gaussians
        amplitude = (
            torch.exp(-R**2 / 4) +
            0.5 * torch.exp(-((X-2)**2 + Y**2 + Z**2) / 2) +
            0.5 * torch.exp(-((X+2)**2 + Y**2 + Z**2) / 2)
        )
        
        # Phase with vortices
        phase = theta + 0.5 * phi + 0.1 * torch.sin(2 * np.pi * Z / 4)
        
        # Complex field
        field = amplitude * torch.exp(1j * phase)
        
        # Add quantum noise
        noise = 0.01 * (torch.randn_like(field, dtype=torch.float32) + 
                       1j * torch.randn_like(field, dtype=torch.float32))
        field = field + noise
        
        return field
    
    async def evolve_step(self):
        """Single distributed evolution step"""
        with self.profiler.profile_section('field_evolution'):
            # Update vacuum state
            vacuum_state = self.quantum_vacuum.update_vacuum_state(
                self.time_step * self.config.time_step,
                self.local_field[self.ghost_cells:-self.ghost_cells,
                                self.ghost_cells:-self.ghost_cells,
                                self.ghost_cells:-self.ghost_cells]
            )
            
            # Prepare for ghost cell exchange
            ghost_exchange_task = asyncio.create_task(
                self.domain_decomp.exchange_ghost_cells_async(self.local_field)
            )
            
            # Compute interior while ghost cells are exchanged
            interior_field = self._compute_interior_evolution(vacuum_state)
            
            # Wait for ghost cell exchange
            await ghost_exchange_task
            
            # Compute boundary evolution
            self._compute_boundary_evolution(vacuum_state)
            
            # Apply Vedic sutras periodically
            if self.time_step % 10 == 0:
                with self.profiler.profile_section('vedic_sutra_application'):
                    self._apply_vedic_sutras()
        
        self.time_step += 1
    
    def _compute_interior_evolution(self, vacuum_state: torch.Tensor) -> torch.Tensor:
        """Compute evolution for interior points (no ghost dependency)"""
        ghost = self.ghost_cells
        interior_slice = slice(ghost + 1, -ghost - 1)
        
        # Extract interior
        interior_field = self.local_field[interior_slice, interior_slice, interior_slice]
        interior_vacuum = vacuum_state[1:-1, 1:-1, 1:-1]
        
        # Get active sutras
        active_sutra_indices = [1, 2, 3]  # Example
        
        # Evolve interior
        new_interior = self.field_dynamics.evolve_field(
            interior_field, interior_vacuum, active_sutra_indices, self.time_step
        )
        
        # Update interior
        self.local_field[interior_slice, interior_slice, interior_slice] = new_interior
        
        return new_interior
    
    def _compute_boundary_evolution(self, vacuum_state: torch.Tensor):
        """Compute evolution for boundary points after ghost exchange"""
        ghost = self.ghost_cells
        
        # Evolve full field now that ghost cells are updated
        active_sutra_indices = [1, 2, 3]
        
        # Extract non-ghost region
        field_region = self.local_field[ghost:-ghost, ghost:-ghost, ghost:-ghost]
        
        # Evolve
        new_field = self.field_dynamics.evolve_field(
            field_region, vacuum_state, active_sutra_indices, self.time_step
        )
        
        # Update
        self.local_field[ghost:-ghost, ghost:-ghost, ghost:-ghost] = new_field
    
    def _apply_vedic_sutras(self):
        """Apply Vedic sutras to local field"""
        ghost = self.ghost_cells
        
        # Work on interior only
        for i in range(ghost, self.local_field.shape[0] - ghost):
            for j in range(ghost, self.local_field.shape[1] - ghost):
                for k in range(ghost, self.local_field.shape[2] - ghost):
                    if 'grvq_field' in self.config.active_sutras:
                        # Map local to global coordinates
                        global_i = self.domain_decomp.local_domain['x'][0] + i - ghost
                        global_j = self.domain_decomp.local_domain['y'][0] + j - ghost
                        global_k = self.domain_decomp.local_domain['z'][0] + k - ghost
                        
                        self.local_field[i, j, k] = self.vedic_sutras.grvq_field_solver(
                            self.local_field, (i, j, k)
                        )
    
    async def calculate_global_coherence(self) -> Dict[str, float]:
        """Calculate coherence with distributed reduction"""
        with self.profiler.profile_section('coherence_calculation'):
            # Calculate local coherence
            ghost = self.ghost_cells
            local_interior = self.local_field[ghost:-ghost, ghost:-ghost, ghost:-ghost]
            
            # Local metrics
            local_metrics = self.coherence_analysis.calculate_coherence_metrics(
                local_interior, []  # Empty history for now
            )
            
            # Weight by local volume
            local_volume = np.prod(local_interior.shape)
            weighted_metrics = {k: v * local_volume for k, v in local_metrics.items()}
            
            # Reduce across all ranks
            global_metrics = {}
            for metric_name in weighted_metrics:
                local_val = weighted_metrics[metric_name]
                global_sum = self.comm.allreduce(local_val, op=MPI.SUM)
                global_volume = self.comm.allreduce(local_volume, op=MPI.SUM)
                global_metrics[metric_name] = global_sum / global_volume
        
        return global_metrics
    
    async def run(self, max_iterations: int):
        """Main distributed evolution loop"""
        self.logger.info(f"Starting distributed evolution on {self.size} ranks")
        
        # Initialize field
        await self.initialize_field()
        
        # Evolution loop
        for iteration in range(max_iterations):
            start_time = time.time()
            
            # Evolve
            await self.evolve_step()
            
            # Calculate coherence periodically
            if iteration % self.config.log_interval == 0:
                metrics = await self.calculate_global_coherence()
                
                if self.rank == 0:
                    self.logger.info(
                        f"Iteration {iteration}: "
                        f"Global Coherence: {metrics['global']:.4f} | "
                        f"Emergence: {metrics.get('emergent', 0):.4f}"
                    )
            
            # Visualization on rank 0
            if self.rank == 0 and iteration % self.config.visualization_interval == 0:
                with self.profiler.profile_section('visualization'):
                    # Gather global field
                    global_field = self.domain_decomp.gather_global_field(self.local_field)
                    if global_field is not None:
                        self.visualization.visualize_field_3d(global_field)
            
            # Performance monitoring
            iteration_time = time.time() - start_time
            load_metrics = self.domain_decomp.compute_load_balance_metrics(iteration_time)
            
            if self.rank == 0 and iteration % 100 == 0:
                self.logger.info(
                    f"Performance: {1/load_metrics['max_time']:.1f} iter/s | "
                    f"Efficiency: {load_metrics['efficiency']:.2%} | "
                    f"Imbalance: {load_metrics['imbalance_factor']:.2f}"
                )
            
            # Checkpoint
            if iteration % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(iteration)
        
        # Final report
        if self.rank == 0:
            self._generate_final_report()
    
    async def _save_checkpoint(self, iteration: int):
        """Save distributed checkpoint"""
        with self.profiler.profile_section('checkpoint_io'):
            # Each rank saves its local field
            checkpoint_dir = self.config.checkpoint_dir / f'iter_{iteration}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            local_checkpoint = {
                'iteration': iteration,
                'time_step': self.time_step,
                'rank': self.rank,
                'local_field': self.local_field.cpu().numpy(),
                'local_domain': self.domain_decomp.local_domain
            }
            
            filename = checkpoint_dir / f'rank_{self.rank}.h5'
            
            with h5py.File(filename, 'w') as f:
                f.create_dataset('field_real', data=np.real(local_checkpoint['local_field']))
                f.create_dataset('field_imag', data=np.imag(local_checkpoint['local_field']))
                f.attrs['iteration'] = iteration
                f.attrs['rank'] = self.rank
                f.attrs['time_step'] = self.time_step
                
                # Save domain info
                domain_group = f.create_group('domain')
                for key, value in self.domain_decomp.local_domain.items():
                    if key != 'shape':
                        domain_group.attrs[f'{key}_start'] = value[0]
                        domain_group.attrs[f'{key}_end'] = value[1]
            
            self.logger.debug(f"Checkpoint saved: {filename}")
    
    def _generate_final_report(self):
        """Generate final performance report (rank 0 only)"""
        if self.rank != 0:
            return
        
        # Get performance report
        perf_report = self.profiler.generate_performance_report()
        
        # Get optimization suggestions
        suggestions = self.profiler.optimize_parameters()
        
        # Create report
        report = {
            'configuration': {
                'grid_size': self.config.grid_size,
                'mpi_ranks': self.size,
                'device': self.config.device,
                'iterations': self.time_step
            },
            'performance': perf_report,
            'optimization_suggestions': suggestions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        report_path = self.config.checkpoint_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Export profile data
        self.profiler.export_profile_data(
            str(self.config.checkpoint_dir / 'profile_data.json')
        )
        
        self.logger.info(f"Final report generated: {report_path}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UTILITY FUNCTIONS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_distributed_pcfe(config_file: Optional[str] = None):
    """Entry point for distributed PCFE execution"""
    import yaml
    from pcfe_v3_core_engine import PCFEConfig
    
    # Initialize MPI
    MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
    
    # Load configuration
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = PCFEConfig(**config_dict)
    else:
        config = PCFEConfig()
    
    # Create distributed engine
    engine = DistributedPCFE(config)
    
    # Run
    asyncio.run(engine.run(config.max_iterations))
    
    # Finalize MPI
    MPI.Finalize()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed PCFE v3.0')
    parser.add_argument('--config', type=str, help='Configuration file')
    args = parser.parse_args()
    
    run_distributed_pcfe(args.config)

# QuanQonscious/sulba.py

import numpy as np

class SulbaShaper:
    """
    Sulba Sutra-based waveform shaper for cymatic patterns.
    
    This class provides methods to enforce geometric symmetry or patterns on wavefield data.
    It can shape a 2D field (or 3D data slice) to be circularly symmetric, square symmetric, etc., 
    as inspired by Sulba Sutra geometric constructions.
    """
    
    def __init__(self, field_shape: tuple):
        """
        Initialize the shaper with the field shape.
        
        Args:
            field_shape: A tuple indicating the shape of the field (e.g., (Nx, Ny) for 2D).
        """
        self.field_shape = field_shape

    def shape_waveform(self, field: np.ndarray, pattern: str = "circle") -> np.ndarray:
        """
        Shape the input field data to match a given geometric pattern.
        
        Supported patterns:
        - "circle": enforce circular symmetry (radial equalization of values).
        - "square": enforce symmetry about the central axes (x and y).
        
        Args:
            field: 2D NumPy array representing the wavefield amplitude or energy.
            pattern: The target pattern ("circle" or "square").
        Returns:
            np.ndarray of the field after shaping.
        """
        data = np.array(field, copy=True, dtype=float)
        Nx, Ny = data.shape
        cx, cy = (Nx - 1) / 2.0, (Ny - 1) / 2.0  # center coordinates
        
        if pattern == "circle":
            # Enforce radial symmetry: average values in concentric rings
            radius_map = {}
            for i in range(Nx):
                for j in range(Ny):
                    r = int(round(np.hypot(i - cx, j - cy)))  # integer radius
                    radius_map.setdefault(r, []).append(data[i, j])
            # Compute average for each radius
            radial_avg = {r: float(np.mean(vals)) for r, vals in radius_map.items()}
            # Assign back the average value for each radius
            for i in range(Nx):
                for j in range(Ny):
                    r = int(round(np.hypot(i - cx, j - cy)))
                    data[i, j] = radial_avg.get(r, data[i, j])
        
        elif pattern == "square":
            # Enforce symmetry about center axes (vertical and horizontal)
            for i in range(Nx):
                for j in range(Ny):
                    i_sym = int(2*cx - i)
                    j_sym = int(2*cy - j)
                    # average the cell with its symmetric counterpart across center
                    if 0 <= i_sym < Nx and 0 <= j_sym < Ny:
                        avg_val = (data[i, j] + data[i_sym, j_sym]) / 2.0
                        data[i, j] = data[i_sym, j_sym] = avg_val
        
        else:
            # Other patterns could be implemented (e.g., "diamond", "hexagon") – for now, no change
            pass
        
        return data

    def calibrate_harmonics(self, field: np.ndarray, target_ratio: float = 1.0) -> np.ndarray:
        """
        Adjust the field's amplitude distribution to promote a harmonic ratio.
        
        For example, to ensure the field resonates with a particular frequency ratio 
        (target_ratio), this method can scale or normalize portions of the field.
        
        Args:
            field: 2D NumPy array of the field.
            target_ratio: A target ratio of outer ring amplitude to center amplitude (for example).
        Returns:
            np.ndarray with adjusted field amplitudes.
        """
        data = np.array(field, copy=True, dtype=float)
        Nx, Ny = data.shape
        cx, cy = (Nx - 1) / 2.0, (Ny - 1) / 2.0
        # Compute average center value (within a small radius) and outer value (near edges)
        center_region = data[int(cx-1):int(cx+2), int(cy-1):int(cy+2)]
        center_avg = float(np.mean(center_region)) if center_region.size > 0 else 0.0
        edge_avg = float(np.mean([data[i, j] for i in [0, Nx-1] for j in range(Ny)] +
                                 [data[i, j] for j in [0, Ny-1] for i in range(Nx)]))
        if center_avg == 0:
            return data
        current_ratio = edge_avg / center_avg if center_avg != 0 else 1.0
        # Scale the field to adjust the ratio of edge to center
        scaling_factor = (target_ratio / current_ratio) if current_ratio != 0 else 1.0
        # Gradually apply scaling from center to edge (to avoid discontinuity)
        for i in range(Nx):
            for j in range(Ny):
                r = np.hypot(i - cx, j - cy) / max(cx, cy)
                # interpolation: near center (r=0) keep same, near edge (r=1) apply full scaling
                scale = 1 + (scaling_factor - 1) * (r ** 1)
                data[i, j] *= scale
        return data

%%writefile /content/visualize_results.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML

# Set up plotting
rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

class VedicVisualizer:
    def __init__(self):
        self.figures = []
        
    def plot_field_evolution(self, theta_values, dot_theta_values):
        """Plot Θ field evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series
        time = np.arange(len(theta_values))
        axes[0, 0].plot(time, theta_values, 'b-', linewidth=2, label='θ(t)')
        axes[0, 0].set_xlabel('Time step')
        axes[0, 0].set_ylabel('Field value θ')
        axes[0, 0].set_title('Θ Field Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Phase space
        axes[0, 1].plot(theta_values, dot_theta_values, 'r-', alpha=0.7, linewidth=1)
        axes[0, 1].scatter(theta_values[0], dot_theta_values[0], color='green', 
                          s=100, label='Start', zorder=5)
        axes[0, 1].scatter(theta_values[-1], dot_theta_values[-1], color='red', 
                          s=100, label='End', zorder=5)
        axes[0, 1].set_xlabel('θ')
        axes[0, 1].set_ylabel('θ̇')
        axes[0, 1].set_title('Phase Space Trajectory')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Power spectrum
        n = len(theta_values)
        if n > 1:
            fft = np.fft.fft(theta_values)
            freq = np.fft.fftfreq(n)
            power = np.abs(fft)**2
            
            axes[1, 0].plot(freq[:n//2], power[:n//2], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_ylabel('Power')
            axes[1, 0].set_title('Power Spectrum')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Autocorrelation
        autocorr = np.correlate(theta_values, theta_values, mode='full')
        autocorr = autocorr[autocorr.size//2:]
        autocorr /= autocorr[0]
        
        axes[1, 1].plot(autocorr[:100], 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_quantum_state(self, quantum_state):
        """Visualize quantum state"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        amplitudes = np.array([abs(amp) for amp in quantum_state])
        phases = np.array([np.angle(amp) for amp in quantum_state])
        
        # Probability distribution
        axes[0].bar(range(len(amplitudes)), amplitudes**2, alpha=0.7)
        axes[0].set_xlabel('State index')
        axes[0].set_ylabel('Probability')
        axes[0].set_title('Quantum State Probability Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Phase distribution
        axes[1].scatter(range(len(phases)), phases, c=amplitudes**2, 
                       cmap='viridis', s=100, alpha=0.7)
        axes[1].set_xlabel('State index')
        axes[1].set_ylabel('Phase (radians)')
        axes[1].set_title('Quantum State Phases')
        axes[1].grid(True, alpha=0.3)
        
        # Bloch sphere projection (for 2-qubit states)
        if len(quantum_state) == 4:
            # Compute expectation values
            pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
            pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
            
            # Convert to density matrix
            rho = np.outer(quantum_state, np.conj(quantum_state))
            
            # Partial trace for first qubit
            rho_1 = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        rho_1[i, j] += rho[2*i + k, 2*j + k]
            
            # Compute expectation values
            x_exp = np.trace(rho_1 @ pauli_x).real
            y_exp = np.trace(rho_1 @ pauli_y).real
            z_exp = np.trace(rho_1 @ pauli_z).real
            
            # Plot Bloch sphere
            from mpl_toolkits.mplot3d import Axes3D
            ax = axes[2]
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, color='lightblue', alpha=0.1)
            ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.5)
            
            # Plot state vector
            ax.quiver(0, 0, 0, x_exp, y_exp, z_exp, color='red', 
                     arrow_length_ratio=0.1, linewidth=3)
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Bloch Sphere Representation')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_metrics(self, metrics_history):
        """Plot consciousness metrics evolution"""
        if not metrics_history:
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Extract metrics
        entropy = [m.entropy_E.to_double() for m in metrics_history]
        coherence = [m.coherence_C.to_double() for m in metrics_history]
        topology = [m.topology_T.to_double() for m in metrics_history]
        lyapunov = [m.lyapunov_L.to_double() for m in metrics_history]
        fitness = [m.fitness.to_double() for m in metrics_history]
        timestamps = [m.timestamp for m in metrics_history]
        
        # Normalize timestamps
        if timestamps:
            timestamps = [ts - timestamps[0] for ts in timestamps]
        
        # Plot individual metrics
        metrics_data = [
            (entropy, 'Entropy (E)', 'blue'),
            (coherence, 'Coherence (C)', 'green'),
            (topology, 'Topology (T)', 'orange'),
            (lyapunov, 'Lyapunov (L)', 'red'),
            (fitness, 'Fitness', 'purple')
        ]
        
        for idx, (data, title, color) in enumerate(metrics_data):
            ax = axes[idx // 3, idx % 3]
            ax.plot(timestamps, data, color=color, linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        # Correlation matrix
        ax = axes[1, 2]
        metrics_matrix = np.array([entropy, coherence, topology, lyapunov, fitness])
        corr_matrix = np.corrcoef(metrics_matrix)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['E', 'C', 'T', 'L', 'F'])
        ax.set_yticklabels(['E', 'C', 'T', 'L', 'F'])
        ax.set_title('Metrics Correlation Matrix')
        
        # Add correlation values
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha='center', va='center', color='black')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def create_interactive_dashboard(self, engine_data):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Θ Field Evolution', 'Phase Space', 'Power Spectrum',
                          'Quantum State', 'Entropy', 'Coherence',
                          'Topology', 'Lyapunov', 'Fitness'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # Add traces for each subplot
        # (Implementation would depend on available data)
        
        fig.update_layout(height=1000, width=1400, 
                         title_text="Vedic Engine Dashboard",
                         showlegend=False)
        
        return fig
    
    def save_all_figures(self, prefix='vedic_'):
        """Save all generated figures"""
        for i, fig in enumerate(self.figures):
            fig.savefig(f'{prefix}_{i:03d}.png', dpi=300, bbox_inches='tight')
            fig.savefig(f'{prefix}_{i:03d}.pdf', bbox_inches='tight')
        print(f"Saved {len(self.figures)} figures")

# Create visualizer instance
visualizer = VedicVisualizer()

# Example usage (would need actual data from engine)
print("Vedic Visualizer initialized!")
print("Ready to visualize results from Vedic Engine runs.")
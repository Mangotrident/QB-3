"""
Quantum Bioenergetics Mapping - ENAQT Engine
Core physics simulation for Environment-Assisted Quantum Transport
Based on validated quantum transport theory (Chin et al. 2013, Rebentrost 2009)
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm, eig
from scipy.sparse import csr_matrix, diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ENAQTEngine:
    """
    Environment-Assisted Quantum Transport (ENAQT) simulation engine
    Models quantum coherence effects in biological energy transport
    """
    
    def __init__(self, n_sites: int = 10, coupling_strength: float = 1.0):
        """
        Initialize ENAQT engine
        
        Args:
            n_sites: Number of quantum sites in the network
            coupling_strength: Base coupling strength between sites
        """
        self.n_sites = n_sites
        self.coupling_strength = coupling_strength
        self.hamiltonian = None
        self.lindblad_operators = None
        self.ete_curve = None
        self.optimal_gamma = None
        self.optimal_tau = None
        
    def build_hamiltonian(self, adjacency_matrix: np.ndarray, site_energies: np.ndarray) -> np.ndarray:
        """
        Build quantum Hamiltonian from network topology
        
        Args:
            adjacency_matrix: Network connectivity matrix
            site_energies: Energy levels for each site
            
        Returns:
            Hamiltonian matrix
        """
        n = len(site_energies)
        H = np.zeros((n, n), dtype=complex)
        
        # Diagonal terms: site energies
        np.fill_diagonal(H, site_energies)
        
        # Off-diagonal terms: coupling between connected sites
        for i in range(n):
            for j in range(n):
                if i != j and adjacency_matrix[i, j] > 0:
                    H[i, j] = -self.coupling_strength * adjacency_matrix[i, j]
                    H[j, i] = np.conj(H[i, j])  # Hermitian
                    
        self.hamiltonian = H
        return H
    
    def build_lindblad_operators(self, gamma: float, tau_c: float) -> List[np.ndarray]:
        """
        Build Lindblad operators for decoherence and dissipation
        
        Args:
            gamma: Dephasing rate
            tau_c: Correlation time
            
        Returns:
            List of Lindblad operators
        """
        n = self.n_sites
        lindblad_ops = []
        
        # Dephasing operators (diagonal)
        for i in range(n):
            L = np.zeros((n, n), dtype=complex)
            L[i, i] = np.sqrt(gamma / (2 * tau_c))
            lindblad_ops.append(L)
        
        # Population transfer operators (off-diagonal)
        for i in range(n):
            for j in range(n):
                if i != j:
                    L = np.zeros((n, n), dtype=complex)
                    L[j, i] = np.sqrt(gamma / tau_c)
                    lindblad_ops.append(L)
                    
        self.lindblad_operators = lindblad_ops
        return lindblad_ops
    
    def lindblad_superoperator(self, H: np.ndarray, lindblad_ops: List[np.ndarray]) -> np.ndarray:
        """
        Construct Lindblad superoperator for master equation
        
        Args:
            H: Hamiltonian matrix
            lindblad_ops: List of Lindblad operators
            
        Returns:
            Lindblad superoperator
        """
        n = H.shape[0]
        dim = n * n
        
        # Commutator with Hamiltonian
        super_H = np.kron(np.eye(n), H) - np.kron(H.T, np.eye(n))
        
        # Dissipation terms
        super_D = np.zeros((dim, dim), dtype=complex)
        
        for L in lindblad_ops:
            # L ρ L†
            term1 = np.kron(np.conj(L), L)
            # -1/2 {L†L, ρ}
            term2 = -0.5 * np.kron(np.eye(n), np.conj(L.T) @ L)
            term3 = -0.5 * np.kron(L.T @ np.conj(L), np.eye(n))
            
            super_D += term1 + term2 + term3
            
        return -1j * super_H + super_D
    
    def simulate_dynamics(self, H: np.ndarray, lindblad_ops: List[np.ndarray], 
                         initial_state: np.ndarray, t_span: Tuple[float, float],
                         n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate quantum dynamics using Lindblad master equation
        
        Args:
            H: Hamiltonian
            lindblad_ops: Lindblad operators
            initial_state: Initial density matrix
            t_span: Time span (t_start, t_end)
            n_points: Number of time points
            
        Returns:
            Time points and density matrices
        """
        super_L = self.lindblad_superoperator(H, lindblad_ops)
        
        # Vectorize density matrix
        rho_vec = initial_state.reshape(-1, 1)
        
        # Time evolution
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        def master_equation(t, rho):
            return super_L @ rho
        
        sol = solve_ivp(master_equation, t_span, rho_vec.flatten(), 
                       t_eval=t_eval, method='RK45', rtol=1e-8)
        
        # Reshape back to density matrices
        rho_matrices = sol.y.T.reshape(-1, H.shape[0], H.shape[1])
        
        return sol.t, rho_matrices
    
    def calculate_ete(self, rho_matrices: np.ndarray, target_site: int) -> float:
        """
        Calculate Energy Transfer Efficiency (ETE)
        
        Args:
            rho_matrices: Time evolution of density matrices
            target_site: Index of target site
            
        Returns:
            ETE value
        """
        # Population at target site over time
        target_population = np.real(np.diagonal(rho_matrices)[:, target_site])
        
        # ETE as integrated population at target
        ete = np.trapz(target_population, dx=1.0)  # Normalized time step
        
        return ete
    
    def simulate_ete_curve(self, gamma_range: Tuple[float, float], 
                          tau_c_range: Tuple[float, float],
                          n_gamma: int = 50, n_tau: int = 50) -> Dict:
        """
        Simulate ETE as function of dephasing rate and correlation time
        
        Args:
            gamma_range: Range of dephasing rates
            tau_c_range: Range of correlation times
            n_gamma: Number of gamma points
            n_tau: Number of tau points
            
        Returns:
            Dictionary with ETE surface and optimal parameters
        """
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian first.")
        
        gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
        tau_vals = np.linspace(tau_c_range[0], tau_c_range[1], n_tau)
        
        # Initial state: population at site 0
        rho_0 = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        rho_0[0, 0] = 1.0
        
        # Target site: last site
        target_site = self.n_sites - 1
        
        # Calculate ETE surface
        ete_surface = np.zeros((n_gamma, n_tau))
        
        for i, gamma in enumerate(gamma_vals):
            for j, tau in enumerate(tau_vals):
                lindblad_ops = self.build_lindblad_operators(gamma, tau)
                _, rho_matrices = self.simulate_dynamics(
                    self.hamiltonian, lindblad_ops, rho_0, (0, 10)
                )
                ete_surface[i, j] = self.calculate_ete(rho_matrices, target_site)
        
        # Find optimal parameters
        max_idx = np.unravel_index(np.argmax(ete_surface), ete_surface.shape)
        self.optimal_gamma = gamma_vals[max_idx[0]]
        self.optimal_tau = tau_vals[max_idx[1]]
        self.ete_curve = ete_surface
        
        return {
            'ete_surface': ete_surface,
            'gamma_vals': gamma_vals,
            'tau_vals': tau_vals,
            'optimal_gamma': self.optimal_gamma,
            'optimal_tau': self.optimal_tau,
            'max_ete': ete_surface[max_idx]
        }
    
    def plot_ete_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ENAQT bell curve
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.ete_curve is None:
            raise ValueError("ETE curve not calculated. Call simulate_ete_curve first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot ETE vs gamma for optimal tau
        optimal_tau_idx = np.argmin(np.abs(self.tau_vals - self.optimal_tau))
        ete_vs_gamma = self.ete_curve[:, optimal_tau_idx]
        
        ax.plot(self.gamma_vals, ete_vs_gamma, 'b-', linewidth=2, label='ETE')
        ax.axvline(self.optimal_gamma, color='r', linestyle='--', 
                  label=f'Optimal γ* = {self.optimal_gamma:.3f}')
        ax.axvline(self.optimal_gamma, color='r', linestyle='--', alpha=0.3)
        ax.fill_between(self.gamma_vals, 0, ete_vs_gamma, alpha=0.3)
        
        ax.set_xlabel('Dephasing Rate (γ)', fontsize=12)
        ax.set_ylabel('Energy Transfer Efficiency', fontsize=12)
        ax.set_title('ENAQT Bell Curve: Quantum Coherence Optimization', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_metrics(self) -> Dict:
        """
        Get key quantum metrics
        
        Returns:
            Dictionary with ETE_peak, γ*, τc values
        """
        if self.ete_curve is None:
            raise ValueError("Simulation not run. Call simulate_ete_curve first.")
        
        return {
            'ETE_peak': np.max(self.ete_curve),
            'gamma_star': self.optimal_gamma,
            'tau_c': self.optimal_tau,
            'coherence_quality': self.calculate_coherence_quality()
        }
    
    def calculate_coherence_quality(self) -> float:
        """
        Calculate coherence quality metric
        
        Returns:
            Coherence quality (0-1 scale)
        """
        if self.ete_curve is None:
            return 0.0
        
        # Width of ENAQT peak as coherence measure
        optimal_tau_idx = np.argmin(np.abs(self.tau_vals - self.optimal_tau))
        ete_vs_gamma = self.ete_curve[:, optimal_tau_idx]
        
        # Full width at half maximum
        half_max = np.max(ete_vs_gamma) / 2
        indices = np.where(ete_vs_gamma >= half_max)[0]
        
        if len(indices) > 0:
            width = self.gamma_vals[indices[-1]] - self.gamma_vals[indices[0]]
            # Normalize by optimal gamma
            coherence_quality = min(1.0, width / self.optimal_gamma)
        else:
            coherence_quality = 0.0
            
        return coherence_quality

def create_sample_network(n_sites: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample quantum network for testing
    
    Args:
        n_sites: Number of sites
        
    Returns:
        Adjacency matrix and site energies
    """
    # Create chain network with some cross-links
    adjacency = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        adjacency[i, i+1] = 1.0
        adjacency[i+1, i] = 1.0
    
    # Add some cross-links for coherence
    if n_sites > 5:
        adjacency[0, 3] = 0.5
        adjacency[3, 0] = 0.5
        adjacency[2, 5] = 0.5
        adjacency[5, 2] = 0.5
    
    # Site energies (gradient from high to low)
    site_energies = np.linspace(2.0, 0.0, n_sites)
    
    return adjacency, site_energies

if __name__ == "__main__":
    # Test the ENAQT engine
    engine = ENAQTEngine(n_sites=10)
    
    # Create sample network
    adjacency, energies = create_sample_network(10)
    engine.build_hamiltonian(adjacency, energies)
    
    # Run simulation
    results = engine.simulate_ete_curve((0.01, 5.0), (0.1, 10.0))
    
    # Get metrics
    metrics = engine.get_metrics()
    print("Quantum Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Plot results
    fig = engine.plot_ete_curve("test_ete_curve.png")
    plt.show()

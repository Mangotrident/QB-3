"""
Quantum Bioenergetics Mapping - Edge Sensitivity Analysis
Analyzes sensitivity of quantum transport to network edge perturbations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import networkx as nx
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class EdgeSensitivityAnalyzer:
    """
    Analyzes how quantum transport efficiency changes with network edge perturbations
    Identifies critical edges and pathways in biological networks
    """
    
    def __init__(self):
        """Initialize edge sensitivity analyzer"""
        self.sensitivity_matrix = None
        self.critical_edges = None
        self.edge_importance = None
        
    def compute_edge_sensitivity(self, adjacency_matrix: np.ndarray,
                                 ete_function: callable,
                                 perturbation_strength: float = 0.1) -> Dict:
        """
        Compute sensitivity of ETE to edge perturbations
        
        Args:
            adjacency_matrix: Original network adjacency matrix
            ete_function: Function that computes ETE given adjacency matrix
            perturbation_strength: Strength of edge perturbation
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        n = adjacency_matrix.shape[0]
        
        # Get baseline ETE
        baseline_ete = ete_function(adjacency_matrix)
        
        # Find all edges
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adjacency_matrix[i, j] > 0:
                    edges.append((i, j))
        
        # Compute sensitivity for each edge
        sensitivity_scores = []
        ete_changes = []
        
        for edge in edges:
            i, j = edge
            
            # Perturb edge (reduce strength)
            perturbed_adj = adjacency_matrix.copy()
            perturbed_adj[i, j] *= (1 - perturbation_strength)
            perturbed_adj[j, i] *= (1 - perturbation_strength)
            
            # Compute new ETE
            new_ete = ete_function(perturbed_adj)
            
            # Sensitivity = relative change in ETE
            if baseline_ete > 0:
                sensitivity = abs(baseline_ete - new_ete) / baseline_ete
            else:
                sensitivity = 0.0
            
            sensitivity_scores.append(sensitivity)
            ete_changes.append(new_ete - baseline_ete)
        
        # Store results
        self.edge_importance = pd.DataFrame({
            'edge': edges,
            'sensitivity': sensitivity_scores,
            'ete_change': ete_changes,
            'original_weight': [adjacency_matrix[i, j] for i, j in edges]
        })
        
        # Sort by sensitivity
        self.edge_importance = self.edge_importance.sort_values('sensitivity', ascending=False)
        
        return {
            'baseline_ete': baseline_ete,
            'edge_importance': self.edge_importance,
            'n_edges': len(edges),
            'mean_sensitivity': np.mean(sensitivity_scores),
            'max_sensitivity': np.max(sensitivity_scores)
        }
    
    def analyze_critical_pathways(self, adjacency_matrix: np.ndarray,
                                 gene_mapping: Dict[int, str]) -> Dict:
        """
        Identify critical pathways based on edge sensitivity
        
        Args:
            adjacency_matrix: Network adjacency matrix
            gene_mapping: Mapping from node indices to gene names
            
        Returns:
            Dictionary with critical pathway analysis
        """
        if self.edge_importance is None:
            raise ValueError("Edge sensitivity not computed. Call compute_edge_sensitivity first.")
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Find top critical edges
        top_edges = self.edge_importance.head(min(10, len(self.edge_importance)))
        
        # Analyze pathways
        critical_pathways = []
        
        for _, row in top_edges.iterrows():
            edge = row['edge']
            sensitivity = row['sensitivity']
            
            # Find shortest paths that use this edge
            try:
                # Get all pairs shortest paths
                all_paths = dict(nx.all_pairs_shortest_path(G))
                
                paths_using_edge = []
                for source in all_paths:
                    for target in all_paths[source]:
                        path = all_paths[source][target]
                        if len(path) > 1:
                            # Check if edge is in path
                            for i in range(len(path) - 1):
                                if (path[i], path[i+1]) == edge or (path[i+1], path[i]) == edge:
                                    gene_path = [gene_mapping.get(node, f"Node_{node}") for node in path]
                                    paths_using_edge.append(gene_path)
                                    break
                
                critical_pathways.append({
                    'edge': edge,
                    'genes': [gene_mapping.get(edge[0], f"Node_{edge[0]}"), 
                             gene_mapping.get(edge[1], f"Node_{edge[1]}")],
                    'sensitivity': sensitivity,
                    'pathways_affected': len(set(tuple(p) for p in paths_using_edge)),
                    'sample_pathways': paths_using_edge[:3]  # Show first 3
                })
                
            except Exception as e:
                print(f"Error analyzing pathways for edge {edge}: {e}")
                continue
        
        self.critical_edges = critical_pathways
        
        return {
            'critical_pathways': critical_pathways,
            'total_critical_edges': len(critical_pathways)
        }
    
    def plot_edge_sensitivity_graph(self, adjacency_matrix: np.ndarray,
                                   gene_mapping: Dict[int, str],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot network with edges colored by sensitivity
        
        Args:
            adjacency_matrix: Network adjacency matrix
            gene_mapping: Mapping from node indices to gene names
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.edge_importance is None:
            raise ValueError("Edge sensitivity not computed. Call compute_edge_sensitivity first.")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Get node positions (spring layout for better visualization)
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Prepare edge colors based on sensitivity
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges():
            # Find sensitivity for this edge
            edge_data = self.edge_importance[
                (self.edge_importance['edge'] == edge) | 
                (self.edge_importance['edge'] == (edge[1], edge[0]))
            ]
            
            if len(edge_data) > 0:
                sensitivity = edge_data.iloc[0]['sensitivity']
                edge_colors.append(sensitivity)
                edge_widths.append(1 + sensitivity * 5)  # Scale width by sensitivity
            else:
                edge_colors.append(0)
                edge_widths.append(1)
        
        # Draw the network
        nodes = nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                      node_size=500, ax=ax)
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                      edge_cmap=plt.cm.Reds, width=edge_widths,
                                      ax=ax)
        
        # Draw labels (gene names)
        labels = {i: gene_mapping.get(i, f"Node_{i}") for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add colorbar for edge sensitivity
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                   norm=plt.Normalize(vmin=0, vmax=max(edge_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Edge Sensitivity', rotation=270, labelpad=15)
        
        ax.set_title('Network Edge Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_ranking(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot bar chart of edge sensitivity rankings
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.edge_importance is None:
            raise ValueError("Edge sensitivity not computed. Call compute_edge_sensitivity first.")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top edges for plotting
        top_n = min(20, len(self.edge_importance))
        top_edges = self.edge_importance.head(top_n)
        
        # Create edge labels
        edge_labels = [f"({edge[0]},{edge[1]})" for edge in top_edges['edge']]
        
        # Plot bar chart
        bars = ax.bar(range(top_n), top_edges['sensitivity'], 
                     color='coral', alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Network Edges', fontsize=12)
        ax.set_ylabel('Sensitivity Score', fontsize=12)
        ax.set_title('Edge Sensitivity Ranking', fontsize=14, fontweight='bold')
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(edge_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_sensitivity_report(self) -> str:
        """
        Generate interpretive text for edge sensitivity analysis
        
        Returns:
            Interpretive summary string
        """
        if self.edge_importance is None:
            return "No edge sensitivity data available for interpretation."
        
        # Basic statistics
        total_edges = len(self.edge_importance)
        mean_sensitivity = self.edge_importance['sensitivity'].mean()
        max_sensitivity = self.edge_importance['sensitivity'].max()
        
        # Get top critical edges
        top_critical = self.edge_importance.head(5)
        
        interpretation = f"Edge sensitivity analysis identified {total_edges} network connections. "
        interpretation += f"Average sensitivity is {mean_sensitivity:.4f}, with maximum sensitivity of {max_sensitivity:.4f}. "
        
        if max_sensitivity > 0.1:
            interpretation += "Several edges show high sensitivity, indicating critical pathways for quantum transport. "
            interpretation += "Targeting these edges could significantly impact energy transfer efficiency. "
            
            if len(top_critical) > 0:
                top_edge = top_critical.iloc[0]
                interpretation += f"The most critical edge is ({top_edge['edge'][0]},{top_edge['edge'][1]}) "
                interpretation += f"with sensitivity {top_edge['sensitivity']:.4f}. "
        else:
            interpretation += "All edges show relatively low sensitivity, suggesting robust quantum transport "
            interpretation += "that is resilient to individual edge perturbations."
        
        # Add pathway information if available
        if self.critical_edges and len(self.critical_edges) > 0:
            interpretation += f" {len(self.critical_edges)} critical pathways identified that may be "
            interpretation += "important targets for therapeutic intervention."
        
        return interpretation
    
    def identify_therapeutic_targets(self, threshold: float = 0.05) -> Dict:
        """
        Identify potential therapeutic targets based on edge sensitivity
        
        Args:
            threshold: Sensitivity threshold for target identification
            
        Returns:
            Dictionary with therapeutic target recommendations
        """
        if self.edge_importance is None:
            raise ValueError("Edge sensitivity not computed. Call compute_edge_sensitivity first.")
        
        # Find edges above threshold
        high_sensitivity_edges = self.edge_importance[
            self.edge_importance['sensitivity'] > threshold
        ].copy()
        
        # Classify targets
        targets = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for _, row in high_sensitivity_edges.iterrows():
            sensitivity = row['sensitivity']
            edge = row['edge']
            
            target_info = {
                'edge': edge,
                'sensitivity': sensitivity,
                'ete_impact': row['ete_change'],
                'recommendation': self._generate_target_recommendation(sensitivity)
            }
            
            if sensitivity > 0.1:
                targets['high_priority'].append(target_info)
            elif sensitivity > 0.05:
                targets['medium_priority'].append(target_info)
            else:
                targets['low_priority'].append(target_info)
        
        return {
            'targets': targets,
            'total_targets': len(high_sensitivity_edges),
            'high_priority_count': len(targets['high_priority']),
            'threshold': threshold
        }
    
    def _generate_target_recommendation(self, sensitivity: float) -> str:
        """Generate recommendation for a target based on sensitivity"""
        if sensitivity > 0.15:
            return "Critical target - high therapeutic potential"
        elif sensitivity > 0.1:
            return "High-value target - consider for drug development"
        elif sensitivity > 0.05:
            return "Moderate target - may enhance combination therapies"
        else:
            return "Low-priority target - limited therapeutic impact"

def create_test_ete_function():
    """Create a test ETE function for demonstration"""
    def test_ete(adjacency_matrix):
        # Simple test: ETE proportional to total edge weight and connectivity
        total_weight = np.sum(adjacency_matrix)
        connectivity = np.count_nonzero(adjacency_matrix) / 2
        
        # Add some nonlinearity
        ete = (total_weight * connectivity) ** 0.5 / 100
        return ete
    
    return test_ete

if __name__ == "__main__":
    # Test the edge sensitivity analyzer
    analyzer = EdgeSensitivityAnalyzer()
    
    # Create test network
    n = 8
    adjacency = np.zeros((n, n))
    
    # Create a connected network
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7),
             (0,3), (2,5), (1,4), (3,6)]
    
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = np.random.uniform(0.5, 1.5)
    
    # Gene mapping
    gene_mapping = {i: f"Gene_{i}" for i in range(n)}
    
    # Create test ETE function
    ete_func = create_test_ete_function()
    
    # Compute sensitivity
    results = analyzer.compute_edge_sensitivity(adjacency, ete_func)
    
    print("Edge Sensitivity Analysis Results:")
    print(f"Baseline ETE: {results['baseline_ete']:.4f}")
    print(f"Mean sensitivity: {results['mean_sensitivity']:.4f}")
    print(f"Max sensitivity: {results['max_sensitivity']:.4f}")
    
    # Analyze pathways
    pathway_results = analyzer.analyze_critical_pathways(adjacency, gene_mapping)
    print(f"Critical pathways identified: {pathway_results['total_critical_edges']}")
    
    # Generate report
    report = analyzer.generate_sensitivity_report()
    print(f"\nInterpretation: {report}")

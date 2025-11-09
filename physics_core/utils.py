"""
Quantum Bioenergetics Mapping - Utility Functions
Common utilities for data processing, file I/O, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataProcessor:
    """Utility class for processing omics and mapping data"""
    
    @staticmethod
    def load_omics_data(file_path: str) -> pd.DataFrame:
        """
        Load omics data from CSV file
        
        Args:
            file_path: Path to CSV file with columns: gene, value
            
        Returns:
            Processed DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate columns
            required_columns = ['gene', 'value']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Clean data
            df = df.dropna(subset=required_columns)
            df['gene'] = df['gene'].astype(str).str.strip()
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            # Remove duplicates (keep last)
            df = df.drop_duplicates(subset=['gene'], keep='last')
            
            return df.sort_values('gene')
            
        except Exception as e:
            raise ValueError(f"Error loading omics data: {str(e)}")
    
    @staticmethod
    def load_mapping_data(file_path: str) -> pd.DataFrame:
        """
        Load gene-to-node mapping data from CSV file
        
        Args:
            file_path: Path to CSV file with columns: gene, node
            
        Returns:
            Processed DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate columns
            required_columns = ['gene', 'node']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Clean data
            df = df.dropna(subset=required_columns)
            df['gene'] = df['gene'].astype(str).str.strip()
            df['node'] = pd.to_numeric(df['node'], errors='coerce')
            df = df.dropna(subset=['node'])
            df['node'] = df['node'].astype(int)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['gene'], keep='last')
            
            return df.sort_values('node')
            
        except Exception as e:
            raise ValueError(f"Error loading mapping data: {str(e)}")
    
    @staticmethod
    def create_adjacency_matrix(omics_df: pd.DataFrame, 
                               mapping_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Create adjacency matrix from omics and mapping data
        
        Args:
            omics_df: DataFrame with gene expression values
            mapping_df: DataFrame with gene-to-node mapping
            
        Returns:
            Tuple of (adjacency_matrix, gene_mapping_dict)
        """
        # Merge data
        merged_df = pd.merge(omics_df, mapping_df, on='gene', how='inner')
        
        if len(merged_df) == 0:
            raise ValueError("No matching genes between omics and mapping data")
        
        # Get unique nodes
        nodes = sorted(merged_df['node'].unique())
        n_nodes = len(nodes)
        
        # Create node-to-index mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create gene-to-node mapping for reference
        gene_to_node = dict(zip(merged_df['gene'], merged_df['node']))
        
        # Initialize adjacency matrix
        adjacency = np.zeros((n_nodes, n_nodes))
        
        # Fill adjacency matrix based on gene expression correlations
        # Group by node and calculate correlations
        node_groups = merged_df.groupby('node')
        
        for node_i, group_i in node_groups:
            for node_j, group_j in node_groups:
                if node_i != node_j:
                    # Calculate correlation between gene sets
                    if len(group_i) > 1 and len(group_j) > 1:
                        # Use mean expression values for simplicity
                        expr_i = group_i['value'].values
                        expr_j = group_j['value'].values
                        
                        # Simple correlation-based edge weight
                        if len(expr_i) == len(expr_j):
                            correlation = np.corrcoef(expr_i, expr_j)[0, 1]
                            if not np.isnan(correlation):
                                weight = abs(correlation)
                                idx_i = node_to_idx[node_i]
                                idx_j = node_to_idx[node_j]
                                adjacency[idx_i, idx_j] = weight
                                adjacency[idx_j, idx_i] = weight
        
        return adjacency, gene_to_node
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data using specified method
        
        Args:
            data: Input data array
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized data array
        """
        if method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        elif method == 'zscore':
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return (data - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

class VisualizationUtils:
    """Utility class for creating visualizations"""
    
    @staticmethod
    def create_interactive_ete_plot(gamma_vals: np.ndarray, 
                                   ete_vals: np.ndarray,
                                   optimal_gamma: float) -> go.Figure:
        """
        Create interactive ENAQT bell curve plot
        
        Args:
            gamma_vals: Gamma values
            ete_vals: ETE values
            optimal_gamma: Optimal gamma value
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add ETE curve
        fig.add_trace(go.Scatter(
            x=gamma_vals,
            y=ete_vals,
            mode='lines',
            name='Energy Transfer Efficiency',
            line=dict(color='#00C2CB', width=3),
            hovertemplate='γ: %{x:.3f}<br>ETE: %{y:.4f}<extra></extra>'
        ))
        
        # Add optimal point
        optimal_idx = np.argmin(np.abs(gamma_vals - optimal_gamma))
        fig.add_trace(go.Scatter(
            x=[optimal_gamma],
            y=[ete_vals[optimal_idx]],
            mode='markers',
            name=f'Optimal (γ* = {optimal_gamma:.3f})',
            marker=dict(color='#FFD166', size=10),
            hovertemplate='γ*: %{x:.3f}<br>ETE_peak: %{y:.4f}<extra></extra>'
        ))
        
        # Add vertical line at optimal gamma
        fig.add_vline(
            x=optimal_gamma,
            line_dash="dash",
            line_color="red",
            annotation_text=f"γ* = {optimal_gamma:.3f}"
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'ENAQT Bell Curve: Quantum Coherence Optimization',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial'}
            },
            xaxis_title='Dephasing Rate (γ)',
            yaxis_title='Energy Transfer Efficiency',
            template='plotly_white',
            width=800,
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    @staticmethod
    def create_resilience_heatmap_plotly(ete_data: np.ndarray,
                                        sample_names: List[str],
                                        gamma_values: np.ndarray) -> go.Figure:
        """
        Create interactive resilience heatmap
        
        Args:
            ete_data: ETE surface data
            sample_names: Names of samples
            gamma_values: Gamma values
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=ete_data.T,
            x=sample_names,
            y=[f'γ_{i:.2f}' for i in gamma_values[::5]],  # Sample every 5th for readability
            colorscale='Viridis',
            colorbar=dict(title="ETE"),
            hovertemplate='Sample: %{x}<br>γ: %{y}<br>ETE: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Quantum Resilience Heatmap',
            xaxis_title='Samples',
            yaxis_title='Dephasing Rate',
            width=1000,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_metrics_dashboard(metrics: Dict) -> go.Figure:
        """
        Create dashboard with key metrics
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ETE Peak', 'Optimal Gamma (γ*)', 'Correlation Time (τc)', 'Coherence Quality'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # ETE Peak
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics.get('ETE_peak', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ETE Peak"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "#00C2CB"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ), row=1, col=1)
        
        # Optimal Gamma
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('gamma_star', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "γ*"},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': "#FFD166"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 3], 'color': "yellow"},
                    {'range': [3, 5], 'color': "green"}
                ]
            }
        ), row=1, col=2)
        
        # Correlation Time
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('tau_c', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "τc"},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "#FF6B6B"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "green"}
                ]
            }
        ), row=2, col=1)
        
        # Coherence Quality
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('coherence_quality', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Coherence Quality"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ]
            }
        ), row=2, col=2)
        
        fig.update_layout(
            title='Quantum Bioenergetics Metrics Dashboard',
            height=600
        )
        
        return fig

class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_results(results: Dict, file_path: str) -> None:
        """
        Save results dictionary to JSON file
        
        Args:
            results: Results dictionary
            file_path: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert recursively
        def recursive_convert(item):
            if isinstance(item, dict):
                return {k: recursive_convert(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [recursive_convert(i) for i in item]
            else:
                return convert_numpy(item)
        
        converted_results = recursive_convert(results)
        
        with open(file_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
    
    @staticmethod
    def load_results(file_path: str) -> Dict:
        """
        Load results from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Results dictionary
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def plot_to_base64(fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    @staticmethod
    def create_sample_data(n_genes: int = 100, n_nodes: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sample omics and mapping data for testing
        
        Args:
            n_genes: Number of genes
            n_nodes: Number of nodes
            
        Returns:
            Tuple of (omics_df, mapping_df)
        """
        # Generate gene names
        gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
        
        # Generate expression values
        np.random.seed(42)
        expression_values = np.random.lognormal(0, 1, n_genes)
        
        # Create omics DataFrame
        omics_df = pd.DataFrame({
            'gene': gene_names,
            'value': expression_values
        })
        
        # Generate mapping (random assignment of genes to nodes)
        node_assignments = np.random.randint(0, n_nodes, n_genes)
        mapping_df = pd.DataFrame({
            'gene': gene_names,
            'node': node_assignments
        })
        
        return omics_df, mapping_df

class ValidationUtils:
    """Utility class for validation and quality control"""
    
    @staticmethod
    def validate_omics_data(df: pd.DataFrame) -> Dict:
        """
        Validate omics data format and quality
        
        Args:
            df: Omics DataFrame
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = ['gene', 'value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report['valid'] = False
            report['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'gene' in df.columns:
            non_string_genes = df['gene'].apply(lambda x: not isinstance(x, str)).sum()
            if non_string_genes > 0:
                report['warnings'].append(f"{non_string_genes} non-string gene names found")
        
        if 'value' in df.columns:
            non_numeric_values = pd.to_numeric(df['value'], errors='coerce').isna().sum()
            if non_numeric_values > 0:
                report['valid'] = False
                report['errors'].append(f"{non_numeric_values} non-numeric values found")
        
        # Check for duplicates
        if 'gene' in df.columns:
            duplicates = df['gene'].duplicated().sum()
            if duplicates > 0:
                report['warnings'].append(f"{duplicates} duplicate gene names found")
        
        # Statistics
        if report['valid'] and 'value' in df.columns:
            numeric_values = pd.to_numeric(df['value'], errors='coerce').dropna()
            if len(numeric_values) > 0:
                report['stats'] = {
                    'n_genes': len(df),
                    'n_valid_values': len(numeric_values),
                    'mean_expression': float(numeric_values.mean()),
                    'std_expression': float(numeric_values.std()),
                    'min_expression': float(numeric_values.min()),
                    'max_expression': float(numeric_values.max())
                }
        
        return report
    
    @staticmethod
    def validate_mapping_data(df: pd.DataFrame, max_nodes: int = 100) -> Dict:
        """
        Validate mapping data format and quality
        
        Args:
            df: Mapping DataFrame
            max_nodes: Maximum allowed number of nodes
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = ['gene', 'node']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report['valid'] = False
            report['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check node values
        if 'node' in df.columns:
            try:
                node_values = pd.to_numeric(df['node'], errors='coerce')
                non_numeric_nodes = node_values.isna().sum()
                if non_numeric_nodes > 0:
                    report['valid'] = False
                    report['errors'].append(f"{non_numeric_nodes} non-numeric node values found")
                
                # Check for negative nodes
                negative_nodes = (node_values < 0).sum()
                if negative_nodes > 0:
                    report['valid'] = False
                    report['errors'].append(f"{negative_nodes} negative node values found")
                
                # Check node range
                valid_nodes = node_values.dropna()
                if len(valid_nodes) > 0:
                    max_node = int(valid_nodes.max())
                    min_node = int(valid_nodes.min())
                    if max_node >= max_nodes:
                        report['warnings'].append(f"Node values exceed recommended maximum of {max_nodes}")
                    
                    report['stats'] = {
                        'n_genes': len(df),
                        'n_nodes': len(valid_nodes.unique()),
                        'min_node': min_node,
                        'max_node': max_node
                    }
                    
            except Exception as e:
                report['valid'] = False
                report['errors'].append(f"Error processing node values: {str(e)}")
        
        return report

if __name__ == "__main__":
    # Test utility functions
    print("Testing QBM Utilities...")
    
    # Test data creation
    omics_df, mapping_df = FileUtils.create_sample_data(50, 10)
    print(f"Created sample data: {len(omics_df)} genes, {mapping_df['node'].nunique()} nodes")
    
    # Test data validation
    omics_validation = ValidationUtils.validate_omics_data(omics_df)
    mapping_validation = ValidationUtils.validate_mapping_data(mapping_df)
    
    print("Omics validation:", "✓" if omics_validation['valid'] else "✗")
    print("Mapping validation:", "✓" if mapping_validation['valid'] else "✗")
    
    # Test adjacency matrix creation
    adjacency, gene_mapping = DataProcessor.create_adjacency_matrix(omics_df, mapping_df)
    print(f"Created adjacency matrix: {adjacency.shape}")
    
    print("All utility tests completed successfully!")

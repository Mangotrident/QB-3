"""
Quantum Bioenergetics Mapping - Resilience Analysis
Computes resilience indices and cohort comparisons for biological systems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ResilienceAnalyzer:
    """
    Analyzes quantum resilience in biological systems
    Computes resilience indices and performs cohort comparisons
    """
    
    def __init__(self):
        """Initialize resilience analyzer"""
        self.resilience_data = None
        self.cohort_labels = None
        self.resilience_index = None
        
    def compute_resilience_index(self, ete_data: np.ndarray, 
                                 gamma_values: np.ndarray,
                                 tau_values: np.ndarray) -> np.ndarray:
        """
        Compute resilience index from ETE surface
        
        Args:
            ete_data: ETE surface data
            gamma_values: Dephasing rate values
            tau_values: Correlation time values
            
        Returns:
            Resilience index array
        """
        # Resilience = area under ENAQT curve / optimal ETE
        resilience_scores = []
        
        for i in range(ete_data.shape[0]):
            # Find optimal tau for this sample
            optimal_tau_idx = np.argmax(ete_data[i, :])
            ete_curve = ete_data[i, :]
            
            # Calculate area under curve
            auc = np.trapz(ete_curve, gamma_values)
            
            # Normalize by maximum ETE
            max_ete = np.max(ete_curve)
            
            # Resilience index (0-1 scale)
            if max_ete > 0:
                resilience = auc / (max_ete * (gamma_values[-1] - gamma_values[0]))
            else:
                resilience = 0.0
                
            resilience_scores.append(resilience)
        
        self.resilience_index = np.array(resilience_scores)
        return self.resilience_index
    
    def analyze_cohort_differences(self, resilience_scores: np.ndarray,
                                   cohort_labels: List[str]) -> Dict:
        """
        Analyze differences between healthy and diseased cohorts
        
        Args:
            resilience_scores: Resilience scores for all samples
            cohort_labels: List of cohort labels ('healthy', 'tumor', etc.)
            
        Returns:
            Dictionary with cohort analysis results
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'resilience': resilience_scores,
            'cohort': cohort_labels
        })
        
        # Group statistics
        cohort_stats = df.groupby('cohort')['resilience'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4)
        
        # Statistical tests
        unique_cohorts = df['cohort'].unique()
        if len(unique_cohorts) == 2:
            # Two-sample t-test equivalent
            group1 = df[df['cohort'] == unique_cohorts[0]]['resilience']
            group2 = df[df['cohort'] == unique_cohorts[1]]['resilience']
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1)-1)*group1.var() + 
                                 (len(group2)-1)*group2.var()) / 
                                (len(group1) + len(group2) - 2))
            cohens_d = (group1.mean() - group2.mean()) / pooled_std
            
            # Correlation if we have paired data
            correlation = None
            if len(group1) == len(group2):
                correlation, p_value = pearsonr(group1, group2)
        else:
            cohens_d = None
            correlation = None
        
        self.cohort_labels = cohort_labels
        self.resilience_data = df
        
        return {
            'cohort_stats': cohort_stats,
            'effect_size': cohens_d,
            'correlation': correlation,
            'n_cohorts': len(unique_cohorts)
        }
    
    def create_resilience_heatmap(self, ete_data: np.ndarray,
                                  sample_names: List[str],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap of resilience scores across samples
        
        Args:
            ete_data: ETE surface data for multiple samples
            sample_names: Names of samples
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap data
        resilience_matrix = ete_data.T  # Transpose for better visualization
        
        # Plot heatmap
        sns.heatmap(resilience_matrix, 
                   xticklabels=sample_names,
                   yticklabels=[f'γ_{i}' for i in range(resilience_matrix.shape[0])],
                   cmap='viridis',
                   cbar_kws={'label': 'ETE'},
                   ax=ax)
        
        ax.set_title('Quantum Resilience Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('Dephasing Rate Index', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cohort_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between cohorts
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.resilience_data is None:
            raise ValueError("Cohort analysis not performed. Call analyze_cohort_differences first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plot
        sns.boxplot(data=self.resilience_data, x='cohort', y='resilience', ax=axes[0, 0])
        axes[0, 0].set_title('Resilience Distribution by Cohort')
        axes[0, 0].set_ylabel('Resilience Index')
        
        # Violin plot
        sns.violinplot(data=self.resilience_data, x='cohort', y='resilience', ax=axes[0, 1])
        axes[0, 1].set_title('Resilience Density by Cohort')
        
        # Histogram
        for cohort in self.resilience_data['cohort'].unique():
            subset = self.resilience_data[self.resilience_data['cohort'] == cohort]
            axes[1, 0].hist(subset['resilience'], alpha=0.7, label=cohort, bins=20)
        axes[1, 0].set_title('Resilience Histogram')
        axes[1, 0].set_xlabel('Resilience Index')
        axes[1, 0].legend()
        
        # Scatter plot (if we have paired data)
        unique_cohorts = self.resilience_data['cohort'].unique()
        if len(unique_cohorts) == 2:
            group1 = self.resilience_data[self.resilience_data['cohort'] == unique_cohorts[0]]['resilience']
            group2 = self.resilience_data[self.resilience_data['cohort'] == unique_cohorts[1]]['resilience']
            
            if len(group1) == len(group2):
                axes[1, 1].scatter(group1, group2, alpha=0.7)
                axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
                axes[1, 1].set_xlabel(f'{unique_cohorts[0]} Resilience')
                axes[1, 1].set_ylabel(f'{unique_cohorts[1]} Resilience')
                axes[1, 1].set_title('Paired Sample Comparison')
        else:
            axes[1, 1].text(0.5, 0.5, 'No paired data\navailable', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Paired Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def identify_outliers(self, threshold: float = 2.0) -> Dict:
        """
        Identify outlier samples based on resilience scores
        
        Args:
            threshold: Standard deviation threshold for outliers
            
        Returns:
            Dictionary with outlier information
        """
        if self.resilience_index is None:
            raise ValueError("Resilience index not computed. Call compute_resilience_index first.")
        
        mean_resilience = np.mean(self.resilience_index)
        std_resilience = np.std(self.resilience_index)
        
        outlier_mask = np.abs(self.resilience_index - mean_resilience) > threshold * std_resilience
        outliers = np.where(outlier_mask)[0]
        
        return {
            'outlier_indices': outliers,
            'outlier_scores': self.resilience_index[outliers],
            'mean_resilience': mean_resilience,
            'std_resilience': std_resilience,
            'threshold': threshold
        }
    
    def generate_resilience_report(self) -> str:
        """
        Generate interpretive text for resilience analysis
        
        Returns:
            Interpretive summary string
        """
        if self.resilience_data is None:
            return "No resilience data available for interpretation."
        
        # Basic statistics
        mean_resilience = self.resilience_data['resilience'].mean()
        std_resilience = self.resilience_data['resilience'].std()
        
        # Cohort comparison
        if len(self.resilience_data['cohort'].unique()) == 2:
            cohort_stats = self.resilience_data.groupby('cohort')['resilience'].mean()
            healthy_mean = cohort_stats.get('healthy', 0)
            tumor_mean = cohort_stats.get('tumor', 0)
            
            if healthy_mean > tumor_mean:
                diff_percent = ((healthy_mean - tumor_mean) / healthy_mean) * 100
                interpretation = f"Transport efficiency is {diff_percent:.1f}% lower in tumor samples, suggesting impaired quantum coherence and possible mitochondrial dysfunction. "
                interpretation += "This reduction in energy transport efficiency may indicate compromised cellular energy metabolism."
            else:
                interpretation = "Unexpected resilience pattern detected. Further investigation recommended."
        else:
            interpretation = f"Average resilience index is {mean_resilience:.3f} ± {std_resilience:.3f}. "
            
            if mean_resilience > 0.7:
                interpretation += "High resilience indicates robust quantum transport and efficient energy metabolism."
            elif mean_resilience > 0.4:
                interpretation += "Moderate resilience suggests some quantum transport efficiency but room for improvement."
            else:
                interpretation += "Low resilience indicates significant impairment in quantum transport, potentially reflecting mitochondrial dysfunction or oxidative stress."
        
        return interpretation
    
    def cluster_samples(self, n_clusters: int = 3) -> Dict:
        """
        Cluster samples based on resilience patterns
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with clustering results
        """
        if self.resilience_index is None:
            raise ValueError("Resilience index not computed. Call compute_resilience_index first.")
        
        # Reshape for clustering
        X = self.resilience_index.reshape(-1, 1)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_analysis[f'cluster_{i}'] = {
                'size': np.sum(mask),
                'mean_resilience': np.mean(self.resilience_index[mask]),
                'std_resilience': np.std(self.resilience_index[mask])
            }
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_analysis': cluster_analysis
        }

if __name__ == "__main__":
    # Test the resilience analyzer
    analyzer = ResilienceAnalyzer()
    
    # Generate sample data
    n_samples = 20
    n_gamma_points = 50
    ete_data = np.random.rand(n_samples, n_gamma_points)  # Sample ETE data
    gamma_values = np.linspace(0.01, 5.0, n_gamma_points)
    tau_values = np.linspace(0.1, 10.0, 20)
    
    # Compute resilience
    resilience = analyzer.compute_resilience_index(ete_data, gamma_values, tau_values)
    
    # Analyze cohorts
    cohort_labels = ['healthy'] * 10 + ['tumor'] * 10
    cohort_results = analyzer.analyze_cohort_differences(resilience, cohort_labels)
    
    print("Resilience Analysis Results:")
    print(f"Mean resilience: {np.mean(resilience):.4f}")
    print(f"Cohort effect size: {cohort_results['effect_size']:.4f}")
    
    # Generate report
    report = analyzer.generate_resilience_report()
    print(f"\nInterpretation: {report}")

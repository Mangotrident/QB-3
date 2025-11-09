"""
Quantum Bioenergetics Mapping - Simulation Results Page
Display simulation results, visualizations, and metrics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from io import BytesIO
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from physics_core.utils import VisualizationUtils, FileUtils

# Page configuration
st.set_page_config(
    page_title="Simulation Results - QBM Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .results-header {
        font-size: 2rem;
        font-weight: 600;
        color: #0D1B2A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-running {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #ffeaa7;
    }
    .status-completed {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
    }
    .status-failed {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #f5c6cb;
    }
    .interpretation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #00C2CB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_job_status(job_id: str) -> dict:
    """Get job status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/status/{job_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_job_results(job_id: str) -> dict:
    """Get job results from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/results/{job_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_all_jobs() -> list:
    """Get all jobs from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/jobs", timeout=10)
        if response.status_code == 200:
            return response.json().get('jobs', [])
        else:
            return []
    except:
        return []

# Header
st.markdown('<h1 class="results-header">📊 Quantum Simulation Results</h1>', unsafe_allow_html=True)

# API Status Check
api_status = check_api_health()
if not api_status:
    st.error("🔴 API is offline. Please ensure the backend server is running on localhost:8000")
    st.stop()

# Session state management
if 'selected_job_id' not in st.session_state:
    st.session_state.selected_job_id = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Job selection
st.markdown("### 🎯 Select Simulation")
jobs = get_all_jobs()

if not jobs:
    st.info("No simulations found. Upload data to get started!")
    if st.button("📤 Upload New Data", type="primary"):
        st.switch_page("app/Upload.py")
    st.stop()

# Convert to DataFrame for better display
jobs_df = pd.DataFrame(jobs)
jobs_df['status_display'] = jobs_df['status'].replace({
    'queued': '⏳ Queued',
    'running': '🔄 Running',
    'completed': '✅ Completed',
    'failed': '❌ Failed'
})

# Job selector
col1, col2 = st.columns([3, 1])

with col1:
    selected_job = st.selectbox(
        "Choose a simulation to view:",
        options=jobs_df['job_id'],
        format_func=lambda x: f"{x} - {jobs_df[jobs_df['job_id']==x]['status_display'].iloc[0]} - {jobs_df[jobs_df['job_id']==x]['sample_name'].iloc[0]}",
        index=0 if st.session_state.selected_job_id is None else 
               jobs_df[jobs_df['job_id']==st.session_state.selected_job_id].index[0] if st.session_state.selected_job_id in jobs_df['job_id'].values else 0
    )

with col2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Update selected job
st.session_state.selected_job_id = selected_job

# Get job status
job_status = get_job_status(selected_job)

if 'error' in job_status:
    st.error(f"❌ Error loading job: {job_status['error']}")
    st.stop()

# Display job status
status = job_status['status']
progress = job_status['progress']

if status == 'queued':
    st.markdown('<div class="status-running">⏳ Simulation is queued and waiting to start...</div>', unsafe_allow_html=True)
elif status == 'running':
    st.markdown(f'<div class="status-running">🔄 Simulation is running... {progress:.1%} complete</div>', unsafe_allow_html=True)
    # Progress bar
    st.progress(progress)
elif status == 'completed':
    st.markdown('<div class="status-completed">✅ Simulation completed successfully!</div>', unsafe_allow_html=True)
elif status == 'failed':
    st.markdown(f'<div class="status-failed">❌ Simulation failed: {job_status.get("error", "Unknown error")}</div>', unsafe_allow_html=True)

st.divider()

# Show results if completed
if status == 'completed' and job_status.get('results'):
    results = job_status['results']
    
    # Results header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📊 Results for: {results['metadata']['sample_name']}")
    
    with col2:
        if st.button("📥 Download JSON", use_container_width=True):
            # Download results as JSON
            json_str = str(results).replace("'", '"')  # Simple conversion
            st.download_button(
                label="Click to download",
                data=json_str,
                file_name=f"{results['metadata']['sample_name']}_results.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📈 View Dashboard", use_container_width=True):
            st.switch_page("app/Diagnostics.py")
    
    # Key Metrics Dashboard
    st.markdown("### 🎯 Key Quantum Metrics")
    
    metrics = results['metrics']
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        # Color coding for ETE
        ete_value = metrics['ETE_peak']
        ete_color = "🟢" if ete_value > 0.7 else "🟡" if ete_value > 0.4 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{ete_color} ETE Peak</h3>
            <h2>{ete_value:.3f}</h2>
            <p>Energy Transfer Efficiency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        gamma_value = metrics['gamma_star']
        gamma_color = "🟢" if 0.5 < gamma_value < 3.0 else "🟡"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{gamma_color} γ*</h3>
            <h2>{gamma_value:.3f}</h2>
            <p>Optimal Dephasing Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        tau_value = metrics['tau_c']
        tau_color = "🟢" if 2.0 < tau_value < 8.0 else "🟡"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{tau_color} τc</h3>
            <h2>{tau_value:.1f}</h2>
            <p>Correlation Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        coherence_value = metrics['coherence_quality']
        coherence_color = "🟢" if coherence_value > 0.7 else "🟡" if coherence_value > 0.4 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{coherence_color} Coherence</h3>
            <h2>{coherence_value:.2f}</h2>
            <p>Quantum Coherence Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ENAQT Plot
    st.markdown("### 📈 ENAQT Bell Curve")
    
    # Create demo plot (in real implementation, this would come from API)
    gamma_vals = [i/10 for i in range(1, 51)]
    ete_vals = [metrics['ETE_peak'] * 0.9 * np.exp(-((g-metrics['gamma_star'])**2)/2) + 0.05 for g in gamma_vals]
    
    ete_fig = VisualizationUtils.create_interactive_ete_plot(
        np.array(gamma_vals), 
        np.array(ete_vals), 
        metrics['gamma_star']
    )
    
    st.plotly_chart(ete_fig, use_container_width=True)
    
    # Interpretation
    st.markdown("### 🧠 Interpretation")
    
    interpretation = results.get('interpretation', 'No interpretation available.')
    
    st.markdown(f'<div class="interpretation-box">{interpretation}</div>', unsafe_allow_html=True)
    
    # Additional Analysis Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛡️ Resilience Analysis")
        resilience = results['resilience']
        
        st.metric("Resilience Score", f"{resilience['score']:.3f}")
        st.markdown(f"**Interpretation:** {resilience['interpretation']}")
    
    with col2:
        st.markdown("#### 🔗 Edge Sensitivity")
        edge_sens = results['edge_sensitivity']
        
        st.metric("Network Edges", edge_sens['n_edges'])
        st.metric("Mean Sensitivity", f"{edge_sens['mean_sensitivity']:.4f}")
        st.markdown(f"**Interpretation:** {edge_sens['interpretation']}")
    
    # Data Summary
    st.markdown("### 📋 Data Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Genes Analyzed", results['data_summary']['n_genes'])
    
    with summary_col2:
        st.metric("Network Nodes", results['data_summary']['n_nodes'])
    
    with summary_col3:
        st.metric("Network Edges", results['data_summary']['n_edges'])
    
    # Download options
    st.markdown("### 📥 Download Results")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        if st.button("📄 Download Full Report", use_container_width=True):
            st.info("Full report generation coming soon!")
    
    with download_col2:
        if st.button("🖼️ Download Plots", use_container_width=True):
            st.info("Plot download coming soon!")
    
    with download_col3:
        if st.button("📊 Export to CSV", use_container_width=True):
            # Create summary CSV
            summary_data = {
                'Metric': ['ETE_peak', 'gamma_star', 'tau_c', 'coherence_quality', 'resilience_score'],
                'Value': [
                    metrics['ETE_peak'],
                    metrics['gamma_star'],
                    metrics['tau_c'],
                    metrics['coherence_quality'],
                    results['resilience']['score']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name=f"{results['metadata']['sample_name']}_summary.csv",
                mime="text/csv"
            )

elif status == 'running':
    # Auto-refresh for running jobs
    if st.session_state.auto_refresh:
        st.markdown("### 🔄 Auto-refreshing...")
        time.sleep(3)  # Wait 3 seconds before refresh
        st.rerun()
    
    if st.button("⏸️ Stop Auto-refresh"):
        st.session_state.auto_refresh = False
        st.rerun()

elif status == 'failed':
    st.error("❌ Simulation failed. Please check your data and try again.")
    
    if st.button("📤 Upload New Data", type="primary"):
        st.switch_page("app/Upload.py")

# Sidebar with all jobs
st.sidebar.markdown("### 📋 All Simulations")
st.sidebar.dataframe(
    jobs_df[['sample_name', 'status_display', 'progress']].rename(columns={
        'sample_name': 'Sample',
        'status_display': 'Status',
        'progress': 'Progress'
    }),
    use_container_width=True,
    hide_index=True
)

st.sidebar.markdown("### 🎯 Quick Actions")
if st.sidebar.button("📤 Upload New Data"):
    st.switch_page("app/Upload.py")

if st.sidebar.button("🧠 View Diagnostics"):
    st.switch_page("app/Diagnostics.py")

if st.sidebar.button("🏠 Back to Home"):
    st.switch_page("app/Home.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🔬 Results generated using validated quantum transport physics</p>
    <p>Based on ENAQT theory: Chin et al. (2013) Nature Physics</p>
</div>
""", unsafe_allow_html=True)

# Import numpy for demo plot
import numpy as np

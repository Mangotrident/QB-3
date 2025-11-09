"""
Quantum Bioenergetics Mapping - Home Page
Main landing page with overview and navigation
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from physics_core.utils import FileUtils, VisualizationUtils

# Page configuration
st.set_page_config(
    page_title="QBM Platform - Quantum Bioenergetics Mapping",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0D1B2A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #00C2CB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #00C2CB;
        margin-bottom: 1rem;
    }
    .highlight {
        color: #FFD166;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚛️ Quantum Bioenergetics Mapping Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Quantum Transport Analysis for Biological Systems</p>', unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# API Status
api_status = check_api_health()
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if api_status:
        st.success("🟢 API Status: Online")
    else:
        st.error("🔴 API Status: Offline")

with status_col2:
    try:
        if api_status:
            response = requests.get(f"{API_BASE_URL}/api/health")
            active_jobs = response.json().get('active_jobs', 0)
            st.info(f"🔄 Active Jobs: {active_jobs}")
        else:
            st.info("🔄 Active Jobs: N/A")
    except:
        st.info("🔄 Active Jobs: N/A")

with status_col3:
    st.info("📊 Platform Version: 1.0.0")

st.divider()

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🧬 What is Quantum Bioenergetics Mapping?")
    
    st.markdown("""
    <div class="feature-card">
        <h4>🔬 Physics-Based Analysis</h4>
        <p>Our platform applies validated quantum transport theory to analyze energy flow in biological systems, 
        providing insights into mitochondrial function and cellular metabolism that traditional methods cannot capture.</p>
    </div>
    
    <div class="feature-card">
        <h4>📊 Key Metrics</h4>
        <p>• <span class="highlight">ETE_peak</span>: Maximum Energy Transfer Efficiency<br>
        • <span class="highlight">γ*</span>: Optimal dephasing rate for quantum coherence<br>
        • <span class="highlight">τc</span>: Correlation time indicating system stability<br>
        • <span class="highlight">Resilience Index</span>: System robustness to perturbations</p>
    </div>
    
    <div class="feature-card">
        <h4>🎯 Applications</h4>
        <p>• Cancer metabolism research<br>
        • Drug efficacy testing<br>
        • Mitochondrial disease diagnostics<br>
        • Aging and longevity studies</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("## 🚀 Quick Start")
    
    if st.button("📤 Upload Data", type="primary", use_container_width=True):
        st.switch_page("app/Upload.py")
    
    if st.button("📊 View Results", use_container_width=True):
        st.switch_page("app/Simulation.py")
    
    if st.button("🧠 Diagnostics", use_container_width=True):
        st.switch_page("app/Diagnostics.py")
    
    st.markdown("---")
    
    st.markdown("### 📋 Sample Data")
    
    if st.button("📥 Download Sample Files", use_container_width=True):
        # Create and provide sample data
        omics_df, mapping_df = FileUtils.create_sample_data(50, 10)
        
        # Convert to CSV for download
        omics_csv = omics_df.to_csv(index=False).encode('utf-8')
        mapping_csv = mapping_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📄 Sample Omics Data",
            data=omics_csv,
            file_name="sample_omics.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.download_button(
            label="📄 Sample Mapping Data",
            data=mapping_csv,
            file_name="sample_mapping.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()

# Recent Activity Section
st.markdown("## 📈 Recent Activity")

if api_status:
    try:
        # Get recent jobs
        response = requests.get(f"{API_BASE_URL}/api/jobs")
        if response.status_code == 200:
            jobs_data = response.json().get('jobs', [])
            
            if jobs_data:
                # Convert to DataFrame
                jobs_df = pd.DataFrame(jobs_data)
                
                # Show last 5 jobs
                recent_jobs = jobs_df.head(5)
                
                # Format for display
                display_df = recent_jobs[['sample_name', 'status', 'progress', 'created_at']].copy()
                display_df['progress'] = display_df['progress'].apply(lambda x: f"{x:.1%}")
                display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No simulations run yet. Upload data to get started!")
                
    except Exception as e:
        st.error(f"Error loading recent activity: {str(e)}")
else:
    st.warning("API is offline. Cannot load recent activity.")

st.divider()

# Key Metrics Dashboard
st.markdown("## 🎯 Platform Capabilities")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🔬 ENAQT Analysis</h3>
        <p>Environment-Assisted Quantum Transport simulation</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Resilience Mapping</h3>
        <p>Cohort comparison and outlier detection</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🧬 Network Analysis</h3>
        <p>Edge sensitivity and pathway identification</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    st.markdown("""
    <div class="metric-card">
        <h3>📝 Reports</h3>
        <p>Automated diagnostic reports with interpretations</p>
    </div>
    """, unsafe_allow_html=True)

# Interactive Demo Section
st.markdown("## 🎮 Interactive Demo")

with st.expander("🔬 Try a Quick Simulation Demo"):
    st.markdown("""
    Experience the power of quantum bioenergetics analysis with our interactive demo.
    
    **What you'll see:**
    - ENAQT bell curves showing optimal coherence conditions
    - Resilience heatmaps comparing different samples
    - Edge sensitivity analysis identifying critical pathways
    - Automated interpretations with clinical insights
    
    **Ready to analyze your own data?** Click the Upload Data button to get started!
    """)
    
    if st.button("🚀 Launch Demo", type="secondary"):
        # Create demo simulation
        demo_metrics = {
            'ETE_peak': 0.73,
            'gamma_star': 1.45,
            'tau_c': 5.2,
            'coherence_quality': 0.68
        }
        
        # Create demo plots
        gamma_vals = [i/10 for i in range(1, 51)]
        ete_vals = [0.7 * np.exp(-((g-1.5)**2)/2) + 0.1 for g in gamma_vals]
        
        demo_fig = VisualizationUtils.create_interactive_ete_plot(
            np.array(gamma_vals), 
            np.array(ete_vals), 
            demo_metrics['gamma_star']
        )
        
        st.plotly_chart(demo_fig, use_container_width=True)
        
        st.markdown("### 📊 Demo Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ETE Peak", f"{demo_metrics['ETE_peak']:.3f}")
        with col2:
            st.metric("γ*", f"{demo_metrics['gamma_star']:.3f}")
        with col3:
            st.metric("τc", f"{demo_metrics['tau_c']:.1f}")
        with col4:
            st.metric("Coherence", f"{demo_metrics['coherence_quality']:.2f}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🧬 Quantum Bioenergetics Mapping Platform | Powered by validated quantum transport physics</p>
    <p>References: Chin et al. (2013) Nat. Phys., Rebentrost (2009) Phys. Rev. Lett.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation info
st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.info("""
**Pages:**
- 🏠 **Home**: Overview and quick start
- 📤 **Upload**: Submit your data for analysis
- 📊 **Simulation**: View results and visualizations
- 🧠 **Diagnostics**: AI-powered insights
- 📜 **About**: Learn more about the science
""")

st.sidebar.markdown("### 📞 Support")
st.sidebar.markdown("""
Need help? Check our documentation or contact support.

**API Status:** """ + ("🟢 Online" if api_status else "🔴 Offline"))

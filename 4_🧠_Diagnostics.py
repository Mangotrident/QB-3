"""
Quantum Bioenergetics Mapping - Diagnostics Assistant Page
AI-powered insights and comprehensive analysis dashboard
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from physics_core.utils import VisualizationUtils, FileUtils

# Page configuration
st.set_page_config(
    page_title="Diagnostics Assistant - QBM Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .diagnostics-header {
        font-size: 2rem;
        font-weight: 600;
        color: #0D1B2A;
        margin-bottom: 1rem;
    }
    .ai-assistant {
        background: linear-gradient(135deg, #00C2CB 0%, #0D1B2A 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #FFD166;
        margin-bottom: 1rem;
    }
    .warning-card {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #ffeaa7;
        margin-bottom: 1rem;
    }
    .success-card {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
        margin-bottom: 1rem;
    }
    .chat-message {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .chat-message.user {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-message.assistant {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
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
    """Get all completed jobs from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/jobs", timeout=10)
        if response.status_code == 200:
            all_jobs = response.json().get('jobs', [])
            # Filter for completed jobs only
            return [job for job in all_jobs if job['status'] == 'completed']
        else:
            return []
    except:
        return []

def generate_ai_insights(results: Dict) -> Dict[str, str]:
    """Generate AI-powered insights from results"""
    metrics = results['metrics']
    metadata = results['metadata']
    
    insights = {}
    
    # ETE Analysis
    ete = metrics['ETE_peak']
    if ete > 0.7:
        insights['ete'] = "Excellent energy transfer efficiency indicates robust mitochondrial function and healthy cellular metabolism."
    elif ete > 0.4:
        insights['ete'] = "Moderate energy transfer efficiency suggests some mitochondrial stress or early metabolic dysfunction."
    else:
        insights['ete'] = "Low energy transfer efficiency indicates significant mitochondrial impairment, possibly due to disease or environmental stress."
    
    # Coherence Analysis
    coherence = metrics['coherence_quality']
    if coherence > 0.7:
        insights['coherence'] = "High quantum coherence quality suggests optimal environmental conditions for biological energy transport."
    elif coherence > 0.4:
        insights['coherence'] = "Moderate coherence quality may indicate suboptimal cellular environment or mild oxidative stress."
    else:
        insights['coherence'] = "Reduced coherence quality suggests significant disruption in quantum transport, possibly from oxidative damage or pathology."
    
    # Sample Type Specific
    sample_type = metadata.get('sample_type', 'unknown')
    if sample_type == 'tumor':
        if ete < 0.5:
            insights['cancer'] = "Reduced efficiency is consistent with Warburg effect and metabolic reprogramming in cancer cells."
        else:
            insights['cancer'] = "Unexpectedly high efficiency may indicate effective metabolic adaptation or treatment response."
    elif sample_type == 'healthy':
        if ete < 0.5:
            insights['health'] = "Lower than expected efficiency may warrant clinical investigation for early disease detection."
        else:
            insights['health'] = "Efficiency is within normal range for healthy tissue."
    
    # Treatment Recommendations
    if ete < 0.4:
        insights['treatment'] = "Consider mitochondrial support therapies: CoQ10, NAD+ precursors, or antioxidant treatments."
    elif metrics['gamma_star'] > 3.0:
        insights['treatment'] = "High dephasing rate suggests sensitivity to environmental factors - consider stress reduction protocols."
    else:
        insights['treatment'] = "Current parameters appear optimal - focus on maintaining existing metabolic health."
    
    return insights

def answer_quantum_question(question: str, context: Dict) -> str:
    """Answer questions about quantum bioenergetics"""
    question_lower = question.lower()
    
    if 'gamma' in question_lower or 'dephasing' in question_lower:
        return """**γ* (Optimal Dephasing Rate)** represents the sweet spot where quantum coherence and environmental noise work together to enhance energy transport.

- **Low γ* (< 0.5)**: Too little quantum effects, transport is classical
- **Optimal γ* (0.5-3.0)**: Perfect balance - ENAQT regime
- **High γ* (> 3.0)**: Too much noise, quantum effects destroyed

Your sample's γ* value indicates how well it maintains quantum coherence during energy transfer."""
    
    elif 'ete' in question_lower or 'efficiency' in question_lower:
        return """**ETE (Energy Transfer Efficiency)** measures how effectively energy moves through the quantum network.

**Scale Interpretation:**
- **> 0.7**: Excellent - robust mitochondrial function
- **0.4-0.7**: Moderate - some impairment or stress
- **< 0.4**: Low - significant mitochondrial dysfunction

ETE integrates quantum coherence, network connectivity, and environmental factors into a single metric of biological energy transport effectiveness."""
    
    elif 'tau' in question_lower or 'correlation' in question_lower:
        return """**τc (Correlation Time)** reflects how long quantum correlations persist in the system.

**Interpretation:**
- **Short τc (< 2)**: Rapid decoherence, classical transport dominates
- **Optimal τc (2-8)**: Balanced quantum-classical transport
- **Long τc (> 8)**: Persistent quantum coherence, highly efficient transport

τc is influenced by temperature, molecular environment, and cellular conditions."""
    
    elif 'coherence' in question_lower:
        return """**Quantum Coherence** in biological systems refers to the maintenance of quantum phase relationships during energy transport.

**Key Points:**
- Enables efficient energy transfer in photosynthesis and cellular respiration
- Sensitive to temperature, pH, and oxidative stress
- Measured through coherence quality index (0-1 scale)
- Higher coherence generally indicates healthier mitochondrial function

Coherence allows energy to explore multiple pathways simultaneously, finding the most efficient route."""
    
    elif 'improve' in question_lower or 'enhance' in question_lower:
        return """**Strategies to Improve Quantum Coherence:**

**🧬 Biological Interventions:**
- Antioxidants (reduce decoherence from oxidative stress)
- NAD+ precursors (support mitochondrial function)
- CoQ10 supplementation (enhance electron transport)
- Temperature optimization (maintain physiological range)

**🔬 Environmental Factors:**
- Reduce exposure to electromagnetic interference
- Maintain optimal pH and ionic strength
- Minimize heavy metal contamination
- Ensure adequate oxygenation

**💊 Pharmacological:**
- Metformin (may enhance mitochondrial efficiency)
- Resveratrol (antioxidant properties)
- Alpha-lipoic acid (cofactor for mitochondrial enzymes)

Always consult with healthcare providers before starting new treatments."""
    
    else:
        return """I can help you understand quantum bioenergetics concepts! Ask me about:

- **γ*** (optimal dephasing rate)
- **ETE** (energy transfer efficiency) 
- **τc** (correlation time)
- **Quantum coherence** and its biological significance
- **Treatment recommendations** based on your results
- **Normal ranges** for different tissue types

What specific aspect would you like to explore?"""

# Header
st.markdown('<h1 class="diagnostics-header">🧠 Quantum Diagnostics Assistant</h1>', unsafe_allow_html=True)

# API Status Check
api_status = check_api_health()
if not api_status:
    st.error("🔴 API is offline. Please ensure the backend server is running on localhost:8000")
    st.stop()

# Get completed jobs
completed_jobs = get_all_jobs()

if not completed_jobs:
    st.info("No completed simulations found. Upload and run data to get diagnostics!")
    st.info("💡 Navigate to the 'Upload' page in the sidebar to submit data for analysis.")
    st.stop()

# Job selection
st.markdown("### 🎯 Select Sample for Analysis")
jobs_df = pd.DataFrame(completed_jobs)

selected_job = st.selectbox(
    "Choose a completed simulation:",
    options=jobs_df['job_id'],
    format_func=lambda x: f"{x} - {jobs_df[jobs_df['job_id']==x]['sample_name'].iloc[0]}"
)

# Get results
results = get_job_results(selected_job)

if 'error' in results:
    st.error(f"❌ Error loading results: {results['error']}")
    st.stop()

# Main layout
main_col, assistant_col = st.columns([3, 2])

with main_col:
    # Sample Overview
    st.markdown("### 📋 Sample Overview")
    
    metadata = results['metadata']
    metrics = results['metrics']
    
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    
    with overview_col1:
        st.metric("Sample", metadata['sample_name'])
        st.metric("Type", metadata['sample_type'])
    
    with overview_col2:
        st.metric("Tissue", metadata['tissue'])
        st.metric("ETE Peak", f"{metrics['ETE_peak']:.3f}")
    
    with overview_col3:
        st.metric("γ*", f"{metrics['gamma_star']:.3f}")
        st.metric("Coherence", f"{metrics['coherence_quality']:.2f}")
    
    # AI Insights
    st.markdown("### 🤖 AI-Powered Insights")
    
    insights = generate_ai_insights(results)
    
    # ETE Insight
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"**⚡ Energy Transfer Analysis:**<br>{insights['ete']}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Coherence Insight
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"**🌊 Quantum Coherence:**<br>{insights['coherence']}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Condition-specific insights
    if 'cancer' in insights:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.markdown(f"**🧬 Cancer Metabolism:**<br>{insights['cancer']}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if 'health' in insights:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.markdown(f"**💚 Health Status:**<br>{insights['health']}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Treatment recommendations
    st.markdown("### 💊 Treatment Recommendations")
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"**🎯 Personalized Recommendations:**<br>{insights['treatment']}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparative Analysis
    st.markdown("### 📊 Comparative Analysis")
    
    # Create comparison with reference values
    ref_metrics = {
        'ETE_peak': {'healthy': 0.65, 'tumor': 0.45, 'your': metrics['ETE_peak']},
        'gamma_star': {'healthy': 1.8, 'tumor': 2.5, 'your': metrics['gamma_star']},
        'coherence_quality': {'healthy': 0.75, 'tumor': 0.55, 'your': metrics['coherence_quality']}
    }
    
    comparison_fig = go.Figure()
    
    categories = ['ETE Peak', 'γ*', 'Coherence Quality']
    healthy_vals = [ref_metrics['ETE_peak']['healthy'], ref_metrics['gamma_star']['healthy'], ref_metrics['coherence_quality']['healthy']]
    tumor_vals = [ref_metrics['ETE_peak']['tumor'], ref_metrics['gamma_star']['tumor'], ref_metrics['coherence_quality']['tumor']]
    your_vals = [ref_metrics['ETE_peak']['your'], ref_metrics['gamma_star']['your'], ref_metrics['coherence_quality']['your']]
    
    comparison_fig.add_trace(go.Bar(
        name='Healthy Average',
        x=categories,
        y=healthy_vals,
        marker_color='green',
        opacity=0.7
    ))
    
    comparison_fig.add_trace(go.Bar(
        name='Tumor Average',
        x=categories,
        y=tumor_vals,
        marker_color='red',
        opacity=0.7
    ))
    
    comparison_fig.add_trace(go.Bar(
        name='Your Sample',
        x=categories,
        y=your_vals,
        marker_color='#00C2CB',
        line=dict(width=3)
    ))
    
    comparison_fig.update_layout(
        title='Your Sample vs. Reference Values',
        xaxis_title='Metrics',
        yaxis_title='Value',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(comparison_fig, use_container_width=True)

with assistant_col:
    # AI Assistant
    st.markdown('<div class="ai-assistant">', unsafe_allow_html=True)
    st.markdown("### 🤖 Quantum Guide")
    st.markdown("Ask me anything about quantum bioenergetics!", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user">👤 You: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant">🤖 Quantum Guide: {message["content"]}</div>', unsafe_allow_html=True)
    
    # Question input
    question = st.text_input("Ask about quantum bioenergetics:", key="question_input")
    
    if st.button("🤔 Ask", type="primary") and question:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": question})
        
        # Generate AI response
        response = answer_quantum_question(question, results)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Clear input and rerun
        st.session_state.question_input = ""
        st.rerun()
    
    # Quick questions
    st.markdown("### 🔍 Quick Questions")
    
    quick_questions = [
        "What does γ* mean?",
        "Is my ETE_peak normal?",
        "How can I improve coherence?",
        "What affects correlation time?",
        "Explain quantum coherence"
    ]
    
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            response = answer_quantum_question(q, results)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Reference Information
    st.markdown("### 📚 References")
    st.markdown("""
    **Key Papers:**
    - Chin et al. (2013) Nature Physics
    - Rebentrost et al. (2009) PRL
    - Engel et al. (2007) Nature
    
    **Normal Ranges:**
    - ETE: 0.4-0.8 (tissue dependent)
    - γ*: 0.5-3.0 (optimal ENAQT)
    - τc: 2-8 (physiological)
    - Coherence: 0.3-0.9 (health dependent)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🧠 AI Assistant provides educational information, not medical advice</p>
    <p>Always consult healthcare professionals for medical decisions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("### 🧭 Quick Navigation")
st.sidebar.info("""
**Pages:**
- 🏠 **Home**: Overview and quick start
- 📤 **Upload**: Submit your data for analysis
- 📊 **Simulation**: View results and visualizations
- 🧠 **Diagnostics**: AI-powered insights
- 📜 **About**: Learn more about the science
""")

st.sidebar.markdown("### 📊 Your Results")
if st.sidebar.button("📥 Download Full Report"):
    st.info("Full report generation coming soon!")

st.sidebar.markdown("### 🔬 Advanced Options")
if st.sidebar.button("🧪 Batch Analysis"):
    st.info("Batch analysis coming soon!")
if st.sidebar.button("📈 Export to CSV"):
    st.info("CSV export coming soon!")

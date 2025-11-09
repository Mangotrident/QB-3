"""
Quantum Bioenergetics Mapping - Upload Page
Data upload and metadata collection for simulation
"""

import streamlit as st
import requests
import pandas as pd
import io
import time
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from physics_core.utils import FileUtils, ValidationUtils, DataProcessor

# Page configuration
st.set_page_config(
    page_title="Upload Data - QBM Platform",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .upload-header {
        font-size: 2rem;
        font-weight: 600;
        color: #0D1B2A;
        margin-bottom: 1rem;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metadata-form {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .file-preview {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 200px;
        overflow-y: auto;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #f5c6cb;
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

def validate_and_preview_file(uploaded_file, file_type: str) -> tuple[bool, str, pd.DataFrame]:
    """Validate uploaded file and return preview"""
    try:
        # Read file content
        content = uploaded_file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Validate based on file type
        if file_type == "omics":
            validation = ValidationUtils.validate_omics_data(df)
            required_cols = ['gene', 'value']
        else:  # mapping
            validation = ValidationUtils.validate_mapping_data(df)
            required_cols = ['gene', 'node']
        
        if not validation['valid']:
            return False, "❌ " + "; ".join(validation['errors']), df
        
        # Check for warnings
        if validation['warnings']:
            warning_text = "⚠️ " + "; ".join(validation['warnings'])
        else:
            warning_text = "✅ File validated successfully"
        
        return True, warning_text, df
        
    except Exception as e:
        return False, f"❌ Error reading file: {str(e)}", pd.DataFrame()

def submit_simulation(omics_file, mapping_file, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Submit simulation to API"""
    try:
        files = {
            'omics_file': omics_file,
            'mapping_file': mapping_file
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/simulate",
            files=files,
            data=metadata,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# Header
st.markdown('<h1 class="upload-header">📤 Upload Data for Quantum Analysis</h1>', unsafe_allow_html=True)

# API Status Check
api_status = check_api_health()
if not api_status:
    st.error("🔴 API is offline. Please ensure the backend server is running on localhost:8000")
    st.stop()

# Progress tracking
if 'upload_step' not in st.session_state:
    st.session_state.upload_step = 1
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'job_submitted' not in st.session_state:
    st.session_state.job_submitted = False

# Step indicator
step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    if st.session_state.upload_step >= 1:
        st.success("✅ Step 1: Upload Files")
    else:
        st.info("⏳ Step 1: Upload Files")

with step_col2:
    if st.session_state.upload_step >= 2:
        st.success("✅ Step 2: Metadata")
    else:
        st.info("⏳ Step 2: Metadata")

with step_col3:
    if st.session_state.upload_step >= 3:
        st.success("✅ Step 3: Submit")
    else:
        st.info("⏳ Step 3: Submit")

st.divider()

# Step 1: File Upload
if st.session_state.upload_step == 1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Step 1: Upload Your Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧬 Omics Data (Gene Expression)")
        st.markdown("""
        **Required format:** CSV file with columns:
        - `gene`: Gene identifier (e.g., "BRCA1", "TP53")
        - `value`: Expression value (numeric)
        
        **Example:**
        ```csv
        gene,value
        BRCA1,12.5
        TP53,8.3
        ```
        """)
        
        omics_file = st.file_uploader(
            "Choose omics CSV file",
            type=['csv'],
            key="omics_upload",
            help="Upload your gene expression data"
        )
        
        if omics_file:
            is_valid, message, preview_df = validate_and_preview_file(omics_file, "omics")
            
            if is_valid:
                st.success(message)
                st.session_state.uploaded_files['omics'] = omics_file
                
                with st.expander("👁️ Preview Data"):
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    st.info(f"📊 Total rows: {len(preview_df)}")
            else:
                st.error(message)
    
    with col2:
        st.markdown("#### 🗺️ Gene-to-Node Mapping")
        st.markdown("""
        **Required format:** CSV file with columns:
        - `gene`: Gene identifier (must match omics data)
        - `node`: Network node assignment (integer)
        
        **Example:**
        ```csv
        gene,node
        BRCA1,0
        TP53,1
        ```
        """)
        
        mapping_file = st.file_uploader(
            "Choose mapping CSV file",
            type=['csv'],
            key="mapping_upload",
            help="Upload your gene-to-node mapping"
        )
        
        if mapping_file:
            is_valid, message, preview_df = validate_and_preview_file(mapping_file, "mapping")
            
            if is_valid:
                st.success(message)
                st.session_state.uploaded_files['mapping'] = mapping_file
                
                with st.expander("👁️ Preview Data"):
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    st.info(f"📊 Total rows: {len(preview_df)} | Unique nodes: {preview_df['node'].nunique()}")
            else:
                st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if (len(st.session_state.uploaded_files) == 2 and 
            st.button("📝 Continue to Metadata", type="primary", use_container_width=True)):
            st.session_state.upload_step = 2
            st.rerun()
    
    # Sample data section
    st.markdown("---")
    st.markdown("### 📥 Need Sample Data?")
    
    sample_col1, sample_col2 = st.columns(2)
    
    with sample_col1:
        if st.button("📄 Download Sample Omics", use_container_width=True):
            omics_df, mapping_df = FileUtils.create_sample_data(50, 10)
            csv = omics_df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="sample_omics.csv",
                mime="text/csv"
            )
    
    with sample_col2:
        if st.button("📄 Download Sample Mapping", use_container_width=True):
            omics_df, mapping_df = FileUtils.create_sample_data(50, 10)
            csv = mapping_df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="sample_mapping.csv",
                mime="text/csv"
            )

# Step 2: Metadata Collection
elif st.session_state.upload_step == 2:
    st.markdown('<div class="metadata-form">', unsafe_allow_html=True)
    st.markdown("### 📝 Step 2: Provide Sample Information")
    
    with st.form("metadata_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏥 Basic Information")
            
            sample_name = st.text_input(
                "Sample Name *",
                value="Sample_001",
                help="Unique identifier for your sample"
            )
            
            sample_type = st.selectbox(
                "Sample Type *",
                options=["healthy", "tumor", "diseased", "treated", "control", "other"],
                help="Biological condition of the sample"
            )
            
            tissue = st.selectbox(
                "Tissue/Organ *",
                options=["brain", "liver", "heart", "lung", "kidney", "muscle", "blood", "other"],
                help="Tissue or organ source"
            )
            
            species = st.selectbox(
                "Species",
                options=["human", "mouse", "rat", "other"],
                help="Species of the sample"
            )
        
        with col2:
            st.markdown("#### 🔬 Experimental Details")
            
            drug_tested = st.text_input(
                "Drug/Compound Tested",
                value="none",
                help="Name of drug or compound being tested"
            )
            
            treatment_duration = st.text_input(
                "Treatment Duration",
                value="",
                help="e.g., 24h, 7 days"
            )
            
            experimental_condition = st.text_area(
                "Experimental Conditions",
                value="",
                help="Additional experimental details"
            )
            
            researcher_name = st.text_input(
                "Researcher Name",
                value="",
                help="Your name for reference"
            )
        
        st.markdown("---")
        
        # Form navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.upload_step = 1
                st.rerun()
        
        with col3:
            submitted = st.form_submit_button("🚀 Submit Analysis", type="primary", use_container_width=True)
        
        if submitted:
            # Validation
            if not sample_name or not sample_type or not tissue:
                st.error("Please fill in all required fields (*)")
            else:
                # Store metadata
                st.session_state.metadata = {
                    'sample_name': sample_name,
                    'sample_type': sample_type,
                    'tissue': tissue,
                    'species': species,
                    'drug_tested': drug_tested,
                    'treatment_duration': treatment_duration,
                    'experimental_condition': experimental_condition,
                    'researcher_name': researcher_name
                }
                st.session_state.upload_step = 3
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Submit and Monitor
elif st.session_state.upload_step == 3:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 🚀 Step 3: Submit for Quantum Analysis")
    
    # Show summary
    st.markdown("#### 📋 Submission Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("**Files Uploaded:**")
        st.info(f"🧬 Omics: {st.session_state.uploaded_files['omics'].name}")
        st.info(f"🗺️ Mapping: {st.session_state.uploaded_files['mapping'].name}")
    
    with summary_col2:
        st.markdown("**Sample Information:**")
        metadata = st.session_state.metadata
        st.info(f"📝 Name: {metadata['sample_name']}")
        st.info(f"🏥 Type: {metadata['sample_type']}")
        st.info(f"🧬 Tissue: {metadata['tissue']}")
        if metadata['drug_tested'] != 'none':
            st.info(f"💊 Drug: {metadata['drug_tested']}")
    
    st.markdown("---")
    
    # Submit button
    if not st.session_state.job_submitted:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🧬 Start Quantum Simulation", type="primary", use_container_width=True):
                with st.spinner("🔄 Submitting to quantum analysis engine..."):
                    # Submit to API
                    result = submit_simulation(
                        st.session_state.uploaded_files['omics'],
                        st.session_state.uploaded_files['mapping'],
                        st.session_state.metadata
                    )
                    
                    if 'error' in result:
                        st.error(f"❌ Submission failed: {result['error']}")
                    else:
                        st.session_state.job_submitted = True
                        st.session_state.job_id = result['job_id']
                        st.success(f"✅ Successfully submitted! Job ID: {result['job_id']}")
                        
                        # Auto-redirect after 3 seconds
                        time.sleep(2)
                        st.switch_page("app/Simulation.py")
    
    else:
        st.success("✅ Simulation submitted successfully!")
        
        if st.button("📊 View Results", type="primary", use_container_width=True):
            st.switch_page("app/Simulation.py")
        
        if st.button("📤 Submit Another Sample", use_container_width=True):
            # Reset state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.upload_step = 1
            st.rerun()
    
    # Back button
    if st.button("⬅️ Back to Metadata"):
        st.session_state.upload_step = 2
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("### 📋 Upload Guidelines")
st.sidebar.info("""
**File Requirements:**
- CSV format only
- Required columns must be present
- Gene names must match between files
- No duplicate gene names allowed

**Data Quality:**
- Remove missing values
- Ensure numeric expression values
- Use consistent gene naming
- Minimum 10 genes recommended

**Processing Time:**
- Small datasets (<100 genes): ~2 minutes
- Medium datasets (100-500 genes): ~5 minutes
- Large datasets (>500 genes): ~10+ minutes
""")

st.sidebar.markdown("### 🔍 Data Validation")
st.sidebar.warning("""
The system automatically validates your data for:
- Required column presence
- Data type correctness
- Missing values
- Duplicate entries
- Gene name consistency

Fix any validation errors before submission.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🔬 Your data will be processed using validated quantum transport physics</p>
    <p>All computations follow ENAQT theory (Chin et al. 2013)</p>
</div>
""", unsafe_allow_html=True)

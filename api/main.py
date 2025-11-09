"""
Quantum Bioenergetics Mapping - FastAPI Backend
REST API for quantum transport simulation and analysis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import os
import uuid
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Import physics core modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from physics_core.enaqt_engine import ENAQTEngine, create_sample_network
from physics_core.resilience import ResilienceAnalyzer
from physics_core.edge_sensitivity import EdgeSensitivityAnalyzer, create_test_ete_function
from physics_core.utils import DataProcessor, FileUtils, ValidationUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Bioenergetics Mapping API",
    description="API for quantum transport simulation in biological systems",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
simulation_cache = {}
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models
class SimulationRequest(BaseModel):
    sample_id: str
    metadata: Dict[str, Union[str, int, float]]
    parameters: Optional[Dict[str, Union[str, int, float]]] = {}

class SimulationResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[Dict] = None
    error: Optional[str] = None

# Helper functions
def generate_job_id() -> str:
    """Generate unique job ID"""
    return str(uuid.uuid4())

def save_uploaded_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            content = upload_file.file.read()
            buffer.write(content)
        return destination
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def run_simulation(job_id: str, omics_data: pd.DataFrame, 
                  mapping_data: pd.DataFrame, metadata: Dict) -> Dict:
    """
    Run complete QBM simulation pipeline
    
    Args:
        job_id: Unique job identifier
        omics_data: Gene expression data
        mapping_data: Gene-to-node mapping
        metadata: Sample metadata
        
    Returns:
        Simulation results dictionary
    """
    try:
        logger.info(f"Starting simulation for job {job_id}")
        
        # Update job status
        simulation_cache[job_id]['status'] = 'running'
        simulation_cache[job_id]['progress'] = 0.1
        
        # Step 1: Validate input data
        omics_validation = ValidationUtils.validate_omics_data(omics_data)
        mapping_validation = ValidationUtils.validate_mapping_data(mapping_data)
        
        if not omics_validation['valid'] or not mapping_validation['valid']:
            raise ValueError("Invalid input data")
        
        simulation_cache[job_id]['progress'] = 0.2
        
        # Step 2: Create adjacency matrix
        adjacency_matrix, gene_mapping = DataProcessor.create_adjacency_matrix(
            omics_data, mapping_data
        )
        
        # Create site energies (gradient based on expression)
        n_sites = adjacency_matrix.shape[0]
        site_energies = np.linspace(2.0, 0.0, n_sites)
        
        simulation_cache[job_id]['progress'] = 0.3
        
        # Step 3: Initialize ENAQT engine
        engine = ENAQTEngine(n_sites=n_sites)
        engine.build_hamiltonian(adjacency_matrix, site_energies)
        
        simulation_cache[job_id]['progress'] = 0.4
        
        # Step 4: Run ENAQT simulation
        gamma_range = (0.01, 5.0)
        tau_range = (0.1, 10.0)
        
        ete_results = engine.simulate_ete_curve(
            gamma_range=gamma_range,
            tau_c_range=tau_range,
            n_gamma=30,  # Reduced for faster processing
            n_tau=20
        )
        
        simulation_cache[job_id]['progress'] = 0.6
        
        # Step 5: Get quantum metrics
        metrics = engine.get_metrics()
        
        simulation_cache[job_id]['progress'] = 0.7
        
        # Step 6: Resilience analysis
        resilience_analyzer = ResilienceAnalyzer()
        resilience_scores = resilience_analyzer.compute_resilience_index(
            ete_results['ete_surface'].reshape(1, -1),
            ete_results['gamma_vals'],
            ete_results['tau_vals']
        )
        
        simulation_cache[job_id]['progress'] = 0.8
        
        # Step 7: Edge sensitivity analysis
        edge_analyzer = EdgeSensitivityAnalyzer()
        
        # Create ETE function for edge analysis
        def ete_function(adj_matrix):
            # Simple proxy function using total connectivity
            return np.sum(adj_matrix) / (np.sum(adj_matrix) + 10)
        
        edge_results = edge_analyzer.compute_edge_sensitivity(
            adjacency_matrix, ete_function, perturbation_strength=0.1
        )
        
        simulation_cache[job_id]['progress'] = 0.9
        
        # Step 8: Generate interpretation
        interpretation = generate_interpretation(metrics, metadata)
        
        # Step 9: Create results directory and save plots
        results_dir = f"results/{job_id}"
        FileUtils.ensure_directory(results_dir)
        
        # Save ENAQT plot
        ete_plot_path = f"{results_dir}/ete_curve.png"
        engine.plot_ete_curve(ete_plot_path)
        
        # Save resilience heatmap
        if len(ete_results['ete_surface']) > 1:
            resilience_plot_path = f"{results_dir}/resilience_heatmap.png"
            resilience_analyzer.create_resilience_heatmap(
                ete_results['ete_surface'],
                [metadata.get('sample_name', 'Sample')],
                resilience_plot_path
            )
        
        # Save edge sensitivity plot
        edge_plot_path = f"{results_dir}/edge_sensitivity.png"
        edge_analyzer.plot_sensitivity_ranking(edge_plot_path)
        
        # Compile final results
        results = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'metrics': {
                'ETE_peak': float(metrics['ETE_peak']),
                'gamma_star': float(metrics['gamma_star']),
                'tau_c': float(metrics['tau_c']),
                'coherence_quality': float(metrics['coherence_quality'])
            },
            'resilience': {
                'score': float(resilience_scores[0]) if len(resilience_scores) > 0 else 0.0,
                'interpretation': resilience_analyzer.generate_resilience_report()
            },
            'edge_sensitivity': {
                'n_edges': int(edge_results['n_edges']),
                'mean_sensitivity': float(edge_results['mean_sensitivity']),
                'max_sensitivity': float(edge_results['max_sensitivity']),
                'interpretation': edge_analyzer.generate_sensitivity_report()
            },
            'interpretation': interpretation,
            'plots': {
                'ete_curve': ete_plot_path,
                'resilience_heatmap': f"{results_dir}/resilience_heatmap.png" if len(ete_results['ete_surface']) > 1 else None,
                'edge_sensitivity': edge_plot_path
            },
            'data_summary': {
                'n_genes': len(omics_data),
                'n_nodes': adjacency_matrix.shape[0],
                'n_edges': int(np.count_nonzero(adjacency_matrix) / 2)
            }
        }
        
        # Save results to JSON
        results_file = f"{results_dir}/results.json"
        FileUtils.save_results(results, results_file)
        
        simulation_cache[job_id]['progress'] = 1.0
        simulation_cache[job_id]['status'] = 'completed'
        simulation_cache[job_id]['results'] = results
        
        logger.info(f"Simulation completed for job {job_id}")
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed for job {job_id}: {str(e)}")
        simulation_cache[job_id]['status'] = 'failed'
        simulation_cache[job_id]['error'] = str(e)
        raise

def generate_interpretation(metrics: Dict, metadata: Dict) -> str:
    """Generate interpretive summary of results"""
    
    ete_peak = metrics.get('ETE_peak', 0)
    gamma_star = metrics.get('gamma_star', 0)
    coherence_quality = metrics.get('coherence_quality', 0)
    
    sample_type = metadata.get('sample_type', 'unknown').lower()
    tissue = metadata.get('tissue', 'unknown')
    
    interpretation = f"Quantum Bioenergetics Analysis for {tissue} sample.\n\n"
    
    # ETE interpretation
    if ete_peak > 0.7:
        interpretation += f"Energy Transfer Efficiency (ETE) is high at {ete_peak:.3f}, indicating robust quantum transport. "
    elif ete_peak > 0.4:
        interpretation += f"ETE is moderate at {ete_peak:.3f}, suggesting some impairment in quantum coherence. "
    else:
        interpretation += f"ETE is low at {ete_peak:.3f}, indicating significant disruption in energy transport. "
    
    # Coherence interpretation
    if coherence_quality > 0.7:
        interpretation += "Quantum coherence is well-maintained, suggesting healthy mitochondrial function. "
    elif coherence_quality > 0.4:
        interpretation += "Moderate coherence quality may indicate early mitochondrial stress. "
    else:
        interpretation += "Reduced coherence suggests possible mitochondrial dysfunction or oxidative stress. "
    
    # Sample type specific interpretation
    if sample_type == 'tumor':
        if ete_peak < 0.5:
            interpretation += "The reduced efficiency is consistent with metabolic reprogramming commonly observed in cancer cells. "
        else:
            interpretation += "Surprisingly high efficiency may indicate effective metabolic adaptation. "
    elif sample_type == 'healthy':
        if ete_peak < 0.5:
            interpretation += "Lower than expected efficiency may warrant further clinical investigation. "
        else:
            interpretation += "Efficiency is within expected range for healthy tissue. "
    
    # Recommendations
    interpretation += "\n\nRecommendations: "
    if ete_peak < 0.4:
        interpretation += "Consider interventions to support mitochondrial health (e.g., antioxidants, metabolic modulators). "
    elif gamma_star > 3.0:
        interpretation += "High optimal dephasing rate suggests sensitivity to environmental factors. "
    else:
        interpretation += "Current quantum transport parameters appear optimal. "
    
    interpretation += "\n\nReferences: Chin et al. (2013) Nat. Phys., Rebentrost (2009) Phys. Rev. Lett."
    
    return interpretation

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum Bioenergetics Mapping API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate_quantum_transport(
    background_tasks: BackgroundTasks,
    omics_file: UploadFile = File(...),
    mapping_file: UploadFile = File(...),
    sample_type: str = "unknown",
    tissue: str = "unknown",
    sample_name: str = "sample",
    drug_tested: str = "none"
):
    """
    Submit quantum transport simulation job
    
    Args:
        omics_file: CSV file with gene expression data (gene, value)
        mapping_file: CSV file with gene-to-node mapping (gene, node)
        sample_type: Type of sample (healthy, tumor, etc.)
        tissue: Tissue or organ type
        sample_name: Sample identifier
        drug_tested: Drug or compound being tested
        
    Returns:
        Job submission response
    """
    try:
        # Generate job ID
        job_id = generate_job_id()
        
        # Create temporary directory for files
        temp_dir = f"temp/{job_id}"
        FileUtils.ensure_directory(temp_dir)
        
        # Save uploaded files
        omics_path = save_uploaded_file(omics_file, f"{temp_dir}/omics.csv")
        mapping_path = save_uploaded_file(mapping_file, f"{temp_dir}/mapping.csv")
        
        # Load and validate data
        omics_data = DataProcessor.load_omics_data(omics_path)
        mapping_data = DataProcessor.load_mapping_data(mapping_path)
        
        # Prepare metadata
        metadata = {
            'sample_type': sample_type,
            'tissue': tissue,
            'sample_name': sample_name,
            'drug_tested': drug_tested,
            'files': {
                'omics': omics_path,
                'mapping': mapping_path
            }
        }
        
        # Initialize job in cache
        simulation_cache[job_id] = {
            'status': 'queued',
            'progress': 0.0,
            'results': None,
            'error': None,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        # Run simulation in background
        background_tasks.add_task(run_simulation, job_id, omics_data, mapping_data, metadata)
        
        return SimulationResponse(
            job_id=job_id,
            status="queued",
            message="Simulation job submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error submitting simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of simulation job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status response
    """
    if job_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = simulation_cache[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data['status'],
        progress=job_data['progress'],
        results=job_data.get('results'),
        error=job_data.get('error')
    )

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """
    Get simulation results
    
    Args:
        job_id: Job identifier
        
    Returns:
        Simulation results
    """
    if job_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = simulation_cache[job_id]
    
    if job_data['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Simulation not completed")
    
    return job_data['results']

@app.get("/api/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """
    Download simulation files
    
    Args:
        job_id: Job identifier
        file_type: Type of file to download (results, plots, report)
        
    Returns:
        File download response
    """
    if job_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = simulation_cache[job_id]
    
    if job_data['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Simulation not completed")
    
    results_dir = f"results/{job_id}"
    
    if file_type == "results":
        file_path = f"{results_dir}/results.json"
    elif file_type == "ete_plot":
        file_path = job_data['results']['plots']['ete_curve']
    elif file_type == "resilience_plot":
        file_path = job_data['results']['plots']['resilience_heatmap']
        if file_path is None:
            raise HTTPException(status_code=404, detail="Resilience plot not available")
    elif file_type == "sensitivity_plot":
        file_path = job_data['results']['plots']['edge_sensitivity']
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='application/octet-stream'
    )

@app.get("/api/jobs")
async def list_jobs():
    """List all simulation jobs"""
    jobs = []
    for job_id, job_data in simulation_cache.items():
        jobs.append({
            'job_id': job_id,
            'status': job_data['status'],
            'progress': job_data['progress'],
            'created_at': job_data['created_at'],
            'sample_name': job_data['metadata'].get('sample_name', 'unknown')
        })
    
    return {"jobs": jobs}

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete simulation job and associated files"""
    if job_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # Remove from cache
        del simulation_cache[job_id]
        
        # Remove files (optional)
        temp_dir = f"temp/{job_id}"
        results_dir = f"results/{job_id}"
        
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        
        return {"message": "Job deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in simulation_cache.values() if j['status'] == 'running']),
        "cache_size": len(simulation_cache)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Quantum Bioenergetics Mapping API")
    
    # Create necessary directories
    FileUtils.ensure_directory("temp")
    FileUtils.ensure_directory("results")
    
    logger.info("API startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down Quantum Bioenergetics Mapping API")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

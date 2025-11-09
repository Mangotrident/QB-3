#!/usr/bin/env python3
"""
Quantum Bioenergetics Mapping Platform - Main Runner
Convenient script to start the complete application stack
"""

import os
import sys
import subprocess
import time
import signal
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import fastapi
        import uvicorn
        import numpy
        import scipy
        import pandas
        import plotly
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server started successfully")
                return process
            else:
                print("❌ API server failed to start")
                return None
        except:
            print("❌ API server not responding")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_streamlit_app():
    """Start the Streamlit frontend application"""
    print("🚀 Starting Streamlit frontend...")
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "app/Home.py", 
            "--server.port", "8501", 
            "--server.address", "0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for app to start
        time.sleep(5)
        
        print("✅ Streamlit app started successfully")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return None

def run_docker():
    """Run the application using Docker Compose"""
    print("🐳 Starting with Docker Compose...")
    try:
        result = subprocess.run([
            "docker-compose", "up", "--build"
        ], check=True)
        print("✅ Docker Compose started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker Compose not found. Please install Docker and Docker Compose.")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["results", "temp", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created/verified")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down servers...")
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Quantum Bioenergetics Mapping Platform")
    parser.add_argument("--mode", choices=["local", "docker"], default="local",
                       help="Run mode: local (Python) or docker")
    parser.add_argument("--api-only", action="store_true",
                       help="Start only the API server")
    parser.add_argument("--frontend-only", action="store_true",
                       help="Start only the Streamlit frontend")
    parser.add_argument("--port-api", type=int, default=8000,
                       help="Port for API server (default: 8000)")
    parser.add_argument("--port-frontend", type=int, default=8501,
                       help="Port for Streamlit frontend (default: 8501)")
    
    args = parser.parse_args()
    
    # Print banner
    print("""
    ⚛️  Quantum Bioenergetics Mapping Platform
    ==========================================
    🧬 Bridging Quantum Physics and Biological Systems
    """)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run based on mode
    if args.mode == "docker":
        success = run_docker()
        sys.exit(0 if success else 1)
    
    # Local mode
    api_process = None
    frontend_process = None
    
    try:
        if not args.frontend_only:
            api_process = start_api_server()
            if not api_process:
                print("❌ Failed to start API server")
                sys.exit(1)
        
        if not args.api_only:
            frontend_process = start_streamlit_app()
            if not frontend_process:
                print("❌ Failed to start Streamlit app")
                if api_process:
                    api_process.terminate()
                sys.exit(1)
        
        # Print access information
        print("""
        🌐 Access the application:
        • Frontend: http://localhost:8501
        • API: http://localhost:8000
        • API Docs: http://localhost:8000/docs
        
        Press Ctrl+C to stop all servers.
        """)
        
        # Wait for processes
        processes = [p for p in [api_process, frontend_process] if p]
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # Clean up processes
        if api_process:
            api_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()

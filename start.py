#!/usr/bin/env python3
"""
Veridata - Advertisement Verification Service
Startup Script

This script helps you start the Veridata application easily.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                        🛡️  VERIDATA                          ║
    ║              Advertisement Verification Service              ║
    ║                                                              ║
    ║        Powered by Gemini AI & Sentiment Analysis           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import google.generativeai
        import textblob
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting backend server...")
    
    backend_path = Path("backend/main.py")
    if not backend_path.exists():
        print("❌ Backend file not found! Please ensure backend/main.py exists.")
        return False
    
    try:
        # Start the backend server
        cmd = [sys.executable, str(backend_path)]
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        print("✅ Backend server starting...")
        print("📍 Backend URL: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def open_frontend():
    """Instructions for opening the frontend"""
    print("\n🌐 Frontend Setup:")
    print("To access the Veridata web interface:")
    print("\n1. Open your web browser")
    print("2. Navigate to the 'frontend' folder")
    print("3. Open 'index.html' file")
    print("\nOr serve it locally:")
    print("   cd frontend")
    print("   python -m http.server 3000")
    print("   Then visit: http://localhost:3000")

def main():
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if files exist
    if not Path("backend/main.py").exists():
        print("❌ Backend files not found!")
        print("Please ensure the following structure exists:")
        print("""
        Veridata/
        ├── backend/
        │   └── main.py
        ├── frontend/
        │   ├── index.html
        │   ├── styles.css
        │   └── script.js
        └── requirements.txt
        """)
        return
    
    print("\n🎯 Starting Veridata Application...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    # Give backend time to start
    print("\n⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Frontend instructions
    open_frontend()
    
    print("\n" + "="*60)
    print("🎉 Veridata is ready!")
    print("="*60)
    print("\n📋 What you can do:")
    print("• Verify advertisement text for accuracy")
    print("• Upload files containing ad content")
    print("• Get AI-powered fact-checking results")
    print("• Analyze sentiment and credibility scores")
    print("\n⚠️  Note: Keep this terminal open to keep the backend running")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Keep the script running and monitor the backend
        while True:
            if backend_process.poll() is not None:
                print("\n❌ Backend process has stopped!")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Veridata...")
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
        print("✅ Backend stopped successfully!")
        print("👋 Thank you for using Veridata!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Startup script for the FastAPI server
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting LLaMA Project Scoring API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

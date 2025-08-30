#Udavit Fast-API (Simplified Version)
import io
import json
import os
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from PIL import Image

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="LLaMA Project Scoring API", version="1.0")

# ------------------------
# Response model
# ------------------------
class ScoreOutput(BaseModel):
    project_id: str
    authenticity_confidence: float
    h2_score: float
    final_score: float
    status: str
    explanation: Dict[str, Any]

# ------------------------
# Helper functions
# ------------------------
def compute_h2_score(h2_produced: float, max_h2: float = 100.0) -> float:
    if h2_produced <= 0:
        return 0.0
    return round(min(h2_produced / max_h2, 1.0), 3)

def combine_scores(auth_conf: float, h2_score: float, weights=(0.6, 0.4)) -> float:
    return round((auth_conf*weights[0] + h2_score*weights[1])*10, 3)

def mock_verify_text(text: str) -> Dict[str, Any]:
    """Mock text verification function"""
    return {"confidence": 0.7, "reason": "Mock text verification completed"}

def verify_image(img: Image.Image) -> Dict[str, Any]:
    w, h = img.size
    return {"confidence": 0.6, "reason": "Image verification completed", "size": [w, h]}

# ------------------------
# Endpoints
# ------------------------
@app.post("/submit", response_model=ScoreOutput)
async def submit_project(
    project_id: str = Form(...),
    startup_name: str = Form(...),
    phase: str = Form(...),
    text: str = Form(...),
    h2_produced: float = Form(...),
    file: UploadFile = File(...)
):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    text_result = mock_verify_text(text)
    image_result = verify_image(img)
    authenticity_conf = round((text_result["confidence"] + image_result["confidence"]) / 2, 3)
    h2_score = compute_h2_score(h2_produced)
    final_score = combine_scores(authenticity_conf, h2_score)

    status = (
        "Approved for subsidy release" if final_score >= 6.5 and authenticity_conf >= 0.5
        else "Needs manual review" if authenticity_conf < 0.5
        else "Rejected"
    )

    explanation = {
        "text_check": text_result,
        "image_check": image_result,
        "authenticity_confidence": authenticity_conf,
        "h2_normalized": h2_score,
        "weights_used": {"authenticity": 0.6, "h2": 0.4}
    }

    return ScoreOutput(
        project_id=project_id,
        authenticity_confidence=authenticity_conf,
        h2_score=h2_score,
        final_score=final_score,
        status=status,
        explanation=explanation
    )

@app.get("/submissions")
async def list_submissions():
    return {"count": 0, "submissions": [], "message": "Mock endpoint - no database configured"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running in mock mode",
        "features": {
            "text_verification": "mock",
            "image_verification": "enabled",
            "database": "disabled"
        }
    }

@app.get("/")
async def root():
    return {"message": "LLaMA Project Scoring API (Mock Mode)", "version": "1.0"}

@app.get("/docs")
async def get_docs():
    return {"message": "API documentation available at /docs endpoint"}

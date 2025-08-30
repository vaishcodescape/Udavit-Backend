#Udavit Fast-API 
import io
import json
import os
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore

# ------------------------
# Hugging Face text-generation pipeline
# ------------------------
try:
    generator = pipeline("text-generation", model="sshleifer/tiny-gpt2", device=-1)
except Exception as e:
    print(f"Warning: Could not load text generation model: {e}")
    generator = None

# ------------------------
# Firebase setup
# ------------------------
try:
    if os.path.exists("firebase_key.json"):
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        projects_collection = db.collection("projects")
        firebase_available = True
    else:
        print("Warning: firebase_key.json not found. Firebase features will be disabled.")
        firebase_available = False
        db = None
        projects_collection = None
except Exception as e:
    print(f"Warning: Firebase initialization failed: {e}")
    firebase_available = False
    db = None
    projects_collection = None

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

def llama_verify_text(text: str) -> Dict[str, Any]:
    if generator is None:
        return {"confidence": 0.5, "reason": "Text generation model not available"}
    
    prompt = (
        f"Assess the authenticity of this project:\n{text}\n"
        "Respond with JSON: {\"confidence\": 0..1, \"reason\": \"short explanation\"}"
    )
    try:
        response = generator(prompt, max_length=200, do_sample=False)
        content = response[0]['generated_text']
        start = content.find("{")
        end = content.rfind("}") + 1
        result_json = content[start:end] if start != -1 and end != -1 else '{}'
        result = json.loads(result_json)
        return {"confidence": float(result.get("confidence", 0.5)), "reason": result.get("reason","")}
    except Exception as e:
        return {"confidence": 0.5, "reason": f"Text check failed: {e}"}

def verify_image(img: Image.Image) -> Dict[str, Any]:
    w, h = img.size
    return {"confidence": 0.6, "reason": "Placeholder image verification", "size": [w, h]}

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
    if not firebase_available:
        raise HTTPException(status_code=503, detail="Firebase service unavailable")
    
    existing = projects_collection.document(project_id).get()
    if existing.exists:
        data = existing.to_dict()
        return ScoreOutput(
            project_id=data["project_id"],
            authenticity_confidence=data["authenticity_confidence"],
            h2_score=data["h2_score"],
            final_score=data["final_score"],
            status=data["status"],
            explanation={"reason": "Project already submitted"}
        )

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    text_result = llama_verify_text(text)
    image_result = verify_image(img)
    authenticity_conf = round((text_result["confidence"] + image_result["confidence"]) / 2, 3)
    h2_score = compute_h2_score(h2_produced)
    final_score = combine_scores(authenticity_conf, h2_score)

    status = (
        "Approved for subsidy release" if final_score >= 6.5 and authenticity_conf >= 0.5
        else "Needs manual review" if authenticity_conf < 0.5
        else "Rejected"
    )

    projects_collection.document(project_id).set({
        "project_id": project_id,
        "startup_name": startup_name,
        "phase": phase,
        "text": text,
        "h2_produced": h2_produced,
        "image_filename": file.filename,
        "authenticity_confidence": authenticity_conf,
        "h2_score": h2_score,
        "final_score": final_score,
        "status": status
    })

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
    if not firebase_available:
        raise HTTPException(status_code=503, detail="Firebase service unavailable")
    
    docs = projects_collection.stream()
    submissions = [doc.to_dict() for doc in docs]
    return {"count": len(submissions), "submissions": submissions}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "firebase_available": firebase_available,
        "text_model_available": generator is not None
    }

@app.get("/")
async def root():
    return {"message": "LLaMA Project Scoring API", "version": "1.0"}

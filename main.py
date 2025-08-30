# Udavit Fast-API 
import io
import os
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from PIL import Image
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------------
# Security
# ------------------------
security = HTTPBearer()

# ------------------------
# Hugging Face text-generation pipeline
# ------------------------
try:
    generator = pipeline("text-generation", model="sshleifer/tiny-gpt2", device=-1)
except Exception as e:
    print(f"⚠️ Warning: Could not load text generation model: {e}")
    generator = None

# ------------------------
# Firebase setup
# ------------------------
try:
    required_vars = [
        "FIREBASE_TYPE",
        "FIREBASE_PROJECT_ID",
        "FIREBASE_PRIVATE_KEY_ID",
        "FIREBASE_PRIVATE_KEY",
        "FIREBASE_CLIENT_EMAIL",
        "FIREBASE_CLIENT_ID",
        "FIREBASE_AUTH_URI",
        "FIREBASE_TOKEN_URI",
        "FIREBASE_AUTH_PROVIDER_X509_CERT_URL",
        "FIREBASE_CLIENT_X509_CERT_URL"
    ]

    if all(os.getenv(var) for var in required_vars):
        cred_dict = {
            "type": os.getenv("FIREBASE_TYPE"),
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
            "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        }

        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        projects_collection = db.collection("projects")
        users_collection = db.collection("users")
        firebase_available = True
        print("✅ Firebase initialized successfully from environment variables")
    else:
        print("⚠️ Firebase environment variables not found. Firebase features will be disabled.")
        firebase_available = False
        db = None
        projects_collection = None
        users_collection = None
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    firebase_available = False
    db = None
    projects_collection = None
    users_collection = None

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="LLaMA Project Scoring API", version="1.0")

# ------------------------
# Response models
# ------------------------
class ScoreOutput(BaseModel):
    project_id: str
    authenticity_confidence: float
    h2_score: float
    final_score: float
    status: str
    explanation: Dict[str, Any]

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    display_name: str
    chemical_industry_role: str
    hydrogen: str
    chemical_company: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    uid: str
    email: str
    display_name: str
    chemical_industry_role: str
    hydrogen: str
    chemical_company: str
    email_verified: bool
    created_at: str

class AuthResponse(BaseModel):
    user: UserResponse
    token: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    oob_code: str
    new_password: str

# ------------------------
# Authentication middleware
# ------------------------
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    if not firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase service unavailable"
        )
    
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(credentials.credentials)
        uid = decoded_token['uid']
        
        # Get user data from Firestore
        user_doc = users_collection.document(uid).get()
        if not user_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_data = user_doc.to_dict()
        user_data['uid'] = uid
        return user_data
        
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )

# ------------------------
# Helper functions
# ------------------------
def compute_h2_score(h2_produced: float, max_h2: float = 100.0) -> float:
    if h2_produced <= 0:
        return 0.0
    return round(min(h2_produced / max_h2, 1.0), 3)

def combine_scores(auth_conf: float, h2_score: float, weights=(0.6, 0.4)) -> float:
    return round((auth_conf * weights[0] + h2_score * weights[1]) * 10, 3)

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
        return {"confidence": float(result.get("confidence", 0.5)), "reason": result.get("reason", "")}
    except Exception as e:
        return {"confidence": 0.5, "reason": f"Text check failed: {e}"}

def verify_image(img: Image.Image) -> Dict[str, Any]:
    w, h = img.size
    return {"confidence": 0.6, "reason": "Placeholder image verification", "size": [w, h]}

# ------------------------
# Authentication endpoints
# ------------------------
@app.post("/auth/register", response_model=AuthResponse)
async def register_user(user_data: UserCreate):
    if not firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase service unavailable"
        )
    
    try:
        # Create user in Firebase Auth
        user_record = auth.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name
        )
        
        # Store additional user data in Firestore
        user_doc_data = {
            "email": user_data.email,
            "display_name": user_data.display_name,
            "chemical_industry_role": user_data.chemical_industry_role,
            "hydrogen": user_data.hydrogen,
            "chemical_company": user_data.chemical_company,
            "email_verified": user_record.email_verified,
            "created_at": user_record.user_metadata.creation_timestamp.isoformat() if user_record.user_metadata.creation_timestamp else None
        }
        
        users_collection.document(user_record.uid).set(user_doc_data)
        
        # Generate custom token for immediate login
        custom_token = auth.create_custom_token(user_record.uid)
        
        # Return user data and token
        return AuthResponse(
            user=UserResponse(
                uid=user_record.uid,
                email=user_record.email,
                display_name=user_record.display_name,
                chemical_industry_role=user_data.chemical_industry_role,
                hydrogen=user_data.hydrogen,
                chemical_company=user_data.chemical_company,
                email_verified=user_record.email_verified,
                created_at=user_doc_data["created_at"] or ""
            ),
            token=custom_token.decode()
        )
        
    except auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/auth/login", response_model=AuthResponse)
async def login_user(login_data: UserLogin):
    if not firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase service unavailable"
        )
    
    try:
        # Get user by email
        user_record = auth.get_user_by_email(login_data.email)
        
        # Get user data from Firestore
        user_doc = users_collection.document(user_record.uid).get()
        if not user_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_data = user_doc.to_dict()
        
        # Generate custom token
        custom_token = auth.create_custom_token(user_record.uid)
        
        return AuthResponse(
            user=UserResponse(
                uid=user_record.uid,
                email=user_record.email,
                display_name=user_data.get("display_name", ""),
                chemical_industry_role=user_data.get("chemical_industry_role", ""),
                hydrogen=user_data.get("hydrogen", ""),
                chemical_company=user_data.get("chemical_company", ""),
                email_verified=user_record.email_verified,
                created_at=user_data.get("created_at", "")
            ),
            token=custom_token.decode()
        )
        
    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/auth/password-reset")
async def request_password_reset(reset_request: PasswordResetRequest):
    if not firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase service unavailable"
        )
    
    try:
        # Generate password reset link
        reset_link = auth.generate_password_reset_link(reset_request.email)
        
        # In a real application, you would send this link via email
        # For now, we'll return it in the response
        return {
            "message": "Password reset link generated successfully",
            "reset_link": reset_link
        }
        
    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset request failed: {str(e)}"
        )

@app.get("/auth/profile", response_model=UserResponse)
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    return UserResponse(
        uid=current_user["uid"],
        email=current_user["email"],
        display_name=current_user["display_name"],
        chemical_industry_role=current_user.get("chemical_industry_role", ""),
        hydrogen=current_user.get("hydrogen", ""),
        chemical_company=current_user.get("chemical_company", ""),
        email_verified=current_user.get("email_verified", False),
        created_at=current_user.get("created_at", "")
    )

@app.put("/auth/profile")
async def update_user_profile(
    display_name: Optional[str] = Form(None),
    chemical_industry_role: Optional[str] = Form(None),
    hydrogen: Optional[str] = Form(None),
    chemical_company: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    if not firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase service unavailable"
        )
    
    try:
        update_data = {}
        if display_name is not None:
            update_data["display_name"] = display_name
        if chemical_industry_role is not None:
            update_data["chemical_industry_role"] = chemical_industry_role
        if hydrogen is not None:
            update_data["hydrogen"] = hydrogen
        if chemical_company is not None:
            update_data["chemical_company"] = chemical_company
        
        if update_data:
            # Update in Firestore
            users_collection.document(current_user["uid"]).update(update_data)
            
            # Update display name in Firebase Auth if provided
            if display_name is not None:
                auth.update_user(
                    current_user["uid"],
                    display_name=display_name
                )
        
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )

@app.delete("/auth/account")
async def delete_user_account(current_user: Dict[str, Any] = Depends(get_current_user)):
    if not firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase service unavailable"
        )
    
    try:
        # Delete user data from Firestore
        users_collection.document(current_user["uid"]).delete()
        
        # Delete user from Firebase Auth
        auth.delete_user(current_user["uid"])
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Account deletion failed: {str(e)}"
        )

# ------------------------
# Protected endpoints (require authentication)
# ------------------------
@app.post("/submit", response_model=ScoreOutput)
async def submit_project(
    project_id: str = Form(...),
    startup_name: str = Form(...),
    phase: str = Form(...),
    text: str = Form(...),
    h2_produced: float = Form(...),
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
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
        "status": status,
        "user_id": current_user["uid"],
        "submitted_at": firestore.SERVER_TIMESTAMP
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
async def list_submissions(current_user: Dict[str, Any] = Depends(get_current_user)):
    if not firebase_available:
        raise HTTPException(status_code=503, detail="Firebase service unavailable")
    
    # Only show user's own submissions
    docs = projects_collection.where("user_id", "==", current_user["uid"]).stream()
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

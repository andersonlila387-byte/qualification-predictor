"""
Smart Applicant Qualification & Adaptability Prediction System
FastAPI Application
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from src.data_preprocessing import (
    ResumeParser, 
    FeatureEngineer, 
    AdaptabilityEvaluator
)
from src.model import QualificationPredictor, initialize_model

# Initialize FastAPI app
app = FastAPI(
    title="Smart Applicant Qualification & Adaptability Prediction System",
    description="ML-powered system that evaluates job applicant suitability through qualifications and adaptability analysis",
    version="1.0.0"
)

# Initialize components
resume_parser = ResumeParser()
feature_engineer = FeatureEngineer()
adaptability_evaluator = AdaptabilityEvaluator()

# Global model variable
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    global predictor
    predictor = QualificationPredictor()
    
    # Try to load existing model or create new one
    if not predictor.load_model():
        print("Training new model...")
        predictor = initialize_model()
        print("Model initialized successfully!")


# Request/Response Models
class JobApplication(BaseModel):
    """Job application details"""
    position: str = Field(..., description="Job position applied for")
    company: str = Field(..., description="Company name")
    required_skills: Optional[List[str]] = Field(default=None, description="Required skills for the position")
    required_experience: Optional[int] = Field(default=None, description="Required years of experience")
    required_education: Optional[int] = Field(default=None, description="Required education level (1-5)")


class AdaptabilityResponse(BaseModel):
    """Adaptability evaluation response"""
    questions: List[dict] = Field(..., description="Tailored follow-up questions")


class AdaptabilitySubmission(BaseModel):
    """Applicant's adaptability response submission"""
    responses: List[str] = Field(..., description="Applicant's responses to adaptability questions")


class PredictionResult(BaseModel):
    """Final prediction result"""
    qualification_probability: float
    adaptability_score: float
    overall_score: float
    decision: str
    recommendation: str


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Applicant Qualification & Adaptability Prediction System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "adaptability_questions": "/adaptability/questions (POST)",
            "evaluate_adaptability": "/adaptability/evaluate (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_trained if predictor else False
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_qualification(
    resume: UploadFile = File(..., description="Resume in PDF format"),
    position: str = Form(..., description="Job position applied for"),
    company: str = Form(..., description="Company name"),
    adaptability_score: Optional[float] = Form(default=0.5, description="Adaptability score (0-1)")
):
    """
    Predict applicant qualification based on resume and job details
    
    Args:
        resume: PDF file of the applicant's resume
        position: Job position applied for
        company: Company name
        adaptability_score: Optional adaptability score (0-1)
    
    Returns:
        Prediction result with qualification probability and decision
    """
    if not predictor or not predictor.is_trained:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Save uploaded PDF temporarily
        temp_pdf_path = f"temp_{resume.filename}"
        with open(temp_pdf_path, "wb") as f:
            content = await resume.read()
            f.write(content)
        
        # Extract features from resume
        features = resume_parser.extract_features(temp_pdf_path)
        
        # Calculate qualification score
        qualification_score = feature_engineer.calculate_qualification_score(
            skill_score=features['skill_score'],
            experience_years=features['experience_years'],
            education_level=features['education_level'],
            job_requirements={}
        )
        
        # Prepare features for ML model
        model_features = feature_engineer.prepare_features_for_model(
            skill_score=features['skill_score'],
            experience_years=features['experience_years'],
            education_level=features['education_level'],
            adaptability_score=adaptability_score
        )
        
        # Get ML model prediction
        ml_result = predictor.predict_single(model_features)
        
        # Calculate overall score (weighted combination)
        overall_score = (
            ml_result['qualification_probability'] * 0.7 +
            adaptability_score * 0.3
        )
        
        # Make final decision
        decision = "Qualified" if overall_score >= 0.6 else "Not Qualified"
        
        # Generate recommendation
        if overall_score >= 0.8:
            recommendation = "Highly recommended for the position"
        elif overall_score >= 0.6:
            recommendation = "Recommended for the position"
        elif overall_score >= 0.4:
            recommendation = "Consider with reservations"
        else:
            recommendation = "Not recommended for the position"
        
        # Clean up temp file
        os.remove(temp_pdf_path)
        
        return PredictionResult(
            qualification_probability=ml_result['qualification_probability'],
            adaptability_score=adaptability_score,
            overall_score=overall_score,
            decision=decision,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/adaptability/questions", response_model=AdaptabilityResponse)
async def get_adaptability_questions(application: JobApplication):
    """
    Get tailored adaptability questions based on job position
    
    Args:
        application: Job application details
    
    Returns:
        List of adaptability evaluation questions
    """
    questions = adaptability_evaluator.get_questions_for_position(application.position)
    
    return AdaptabilityResponse(questions=questions)


@app.post("/adaptability/evaluate")
async def evaluate_adaptability(submission: AdaptabilitySubmission):
    """
    Evaluate applicant adaptability based on responses
    
    Args:
        submission: Applicant's responses to adaptability questions
    
    Returns:
        Adaptability score and evaluation
    """
    score = adaptability_evaluator.calculate_adaptability_score(submission.responses)
    
    return {
        "adaptability_score": score,
        "evaluation": "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low",
        "interpretation": {
            "high": "Applicant demonstrates strong adaptability and learning capabilities",
            "medium": "Applicant shows moderate adaptability potential",
            "low": "Applicant may need additional support for role adaptability"
        }
    }


@app.post("/features/extract")
async def extract_resume_features(resume: UploadFile = File(..., description="Resume in PDF format")):
    """
    Extract features from resume without making prediction
    
    Args:
        resume: PDF file of the applicant's resume
    
    Returns:
        Extracted features from the resume
    """
    try:
        # Save uploaded PDF temporarily
        temp_pdf_path = f"temp_{resume.filename}"
        with open(temp_pdf_path, "wb") as f:
            content = await resume.read()
            f.write(content)
        
        # Extract features from resume
        features = resume_parser.extract_features(temp_pdf_path)
        
        # Clean up temp file
        os.remove(temp_pdf_path)
        
        return {
            "status": "success",
            "features": {
                "skills": features['skills'],
                "skills_count": features['skills_count'],
                "skill_score": round(features['skill_score'], 3),
                "experience_years": features['experience_years'],
                "education_level": features['education_level']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

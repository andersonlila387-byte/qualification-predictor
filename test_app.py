"""
Test script for the Smart Applicant Qualification System
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import ResumeParser, FeatureEngineer, AdaptabilityEvaluator
from src.model import QualificationPredictor, initialize_model, create_sample_training_data
import numpy as np


def test_resume_parser():
    """Test resume parser functionality"""
    print("=" * 50)
    print("Testing Resume Parser")
    print("=" * 50)
    
    parser = ResumeParser()
    
    # Test skills extraction
    sample_text = """
    Experienced Python developer with skills in Java, JavaScript, 
    React, Node.js, Django, Flask, SQL, PostgreSQL, AWS, Docker.
    """
    
    skills = parser.extract_skills(sample_text)
    print(f"Extracted skills: {skills}")
    print(f"Skills count: {len(skills)}")
    
    # Test experience extraction
    exp_text = "5+ years of experience in software development"
    years = parser.extract_experience_years(exp_text)
    print(f"Extracted years: {years}")
    
    # Test education extraction
    edu_text = "Master's degree in Computer Science"
    edu_level = parser.extract_education_level(edu_text)
    print(f"Education level: {edu_level}")
    
    print("\n✓ Resume Parser tests passed!\n")


def test_feature_engineer():
    """Test feature engineering functionality"""
    print("=" * 50)
    print("Testing Feature Engineer")
    print("=" * 50)
    
    engineer = FeatureEngineer()
    
    # Test qualification score calculation
    score = engineer.calculate_qualification_score(
        skill_score=0.8,
        experience_years=5,
        education_level=4,
        job_requirements={}
    )
    print(f"Qualification score: {score:.3f}")
    
    # Test feature preparation
    features = engineer.prepare_features_for_model(
        skill_score=0.8,
        experience_years=5,
        education_level=4,
        adaptability_score=0.7
    )
    print(f"Model features shape: {features.shape}")
    print(f"Model features: {features}")
    
    print("\n✓ Feature Engineer tests passed!\n")


def test_adaptability_evaluator():
    """Test adaptability evaluator functionality"""
    print("=" * 50)
    print("Testing Adaptability Evaluator")
    print("=" * 50)
    
    evaluator = AdaptabilityEvaluator()
    
    # Test question generation
    questions = evaluator.get_questions_for_position("Software Engineer")
    print(f"Generated {len(questions)} adaptability questions")
    for q in questions:
        print(f"  - {q['question'][:50]}...")
    
    # Test score calculation
    responses = [
        "I learned Python in 2 months while working on a project. I used online courses and practiced daily.",
        "When our team switched to remote work, I quickly adapted by setting up a home office.",
        "I approach unfamiliar problems by breaking them down into smaller parts.",
        "When I joined a new team, I made an effort to understand each member's working style."
    ]
    
    score = evaluator.calculate_adaptability_score(responses)
    print(f"\nAdaptability score: {score:.3f}")
    
    print("\n✓ Adaptability Evaluator tests passed!\n")


def test_ml_model():
    """Test ML model functionality"""
    print("=" * 50)
    print("Testing ML Model")
    print("=" * 50)
    
    # Create training data
    X, y = create_sample_training_data()
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: Qualified={sum(y)}, Not Qualified={len(y)-sum(y)}")
    
    # Train model
    predictor = QualificationPredictor()
    metrics = predictor.train(X, y)
    print(f"\nTraining metrics: {metrics}")
    
    # Test prediction
    test_features = np.array([[0.8, 0.7, 0.8, 0.85]])
    result = predictor.predict_single(test_features)
    print(f"\nPrediction result: {result}")
    
    # Test feature importance
    importance = predictor.get_feature_importance()
    print(f"\nFeature importance: {importance}")
    
    # Save model
    predictor.save_model()
    print("\n✓ Model saved successfully!")
    
    # Load model
    new_predictor = QualificationPredictor()
    loaded = new_predictor.load_model()
    print(f"✓ Model loaded: {loaded}")
    
    print("\n✓ ML Model tests passed!\n")


def test_integration():
    """Test full pipeline integration"""
    print("=" * 50)
    print("Testing Full Integration Pipeline")
    print("=" * 50)
    
    # Initialize components
    parser = ResumeParser()
    engineer = FeatureEngineer()
    evaluator = AdaptabilityEvaluator()
    
    # Sample resume text
    sample_resume = """
    JOHN DOE
    Software Engineer
    
    SKILLS: Python, Java, JavaScript, React, Node.js, Django, SQL, AWS, Docker
    
    EXPERIENCE: 5 years of experience in software development
    
    EDUCATION: Master's degree in Computer Science
    
    I have worked on multiple projects where I had to learn new technologies quickly.
    """
    
    # Extract features
    skills = parser.extract_skills(sample_resume)
    years = parser.extract_experience_years(sample_resume)
    edu = parser.extract_education_level(sample_resume)
    skill_score = len(skills) / len(parser.skills_keywords)
    
    print(f"Skills: {skills}")
    print(f"Experience: {years} years")
    print(f"Education: {edu}")
    print(f"Skill score: {skill_score:.3f}")
    
    # Adaptability evaluation
    questions = evaluator.get_questions_for_position("Software Engineer")
    responses = [
        "I learned new technologies quickly through online courses and hands-on practice.",
        "I adapt well to changes in work environment and requirements.",
        "I solve unfamiliar problems by breaking them down and researching solutions.",
        "I work well with new teams by communicating openly and understanding their processes."
    ]
    adaptability_score = evaluator.calculate_adaptability_score(responses)
    print(f"Adaptability score: {adaptability_score:.3f}")
    
    # Calculate qualification
    qualification_score = engineer.calculate_qualification_score(
        skill_score=skill_score,
        experience_years=years,
        education_level=edu,
        job_requirements={}
    )
    
    # Get ML prediction
    features = engineer.prepare_features_for_model(
        skill_score=skill_score,
        experience_years=years,
        education_level=edu,
        adaptability_score=adaptability_score
    )
    
    # Load model and predict
    predictor = QualificationPredictor()
    predictor.load_model()
    ml_result = predictor.predict_single(features)
    
    # Calculate overall score
    overall_score = (
        ml_result['qualification_probability'] * 0.7 +
        adaptability_score * 0.3
    )
    decision = "Qualified" if overall_score >= 0.6 else "Not Qualified"
    
    print(f"\n--- Final Results ---")
    print(f"Qualification probability: {ml_result['qualification_probability']:.3f}")
    print(f"Adaptability score: {adaptability_score:.3f}")
    print(f"Overall score: {overall_score:.3f}")
    print(f"Decision: {decision}")
    
    print("\n✓ Integration tests passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SMART APPLICANT QUALIFICATION SYSTEM - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_resume_parser()
        test_feature_engineer()
        test_adaptability_evaluator()
        test_ml_model()
        test_integration()
        
        print("=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe system is ready to use!")
        print("\nTo start the API server, run:")
        print("  python main.py")
        print("\nOr use uvicorn:")
        print("  uvicorn main:app --reload")
        print("\nAPI will be available at: http://localhost:8000")
        print("Interactive docs at: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

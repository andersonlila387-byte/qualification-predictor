"""
Data Preprocessing Module
Handles resume parsing and feature extraction for the Smart Applicant Qualification System
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from PyPDF2 import PdfReader


class ResumeParser:
    """Parses resume PDF and extracts key features"""
    
    def __init__(self):
        self.skills_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'git', 'jira', 'agile', 'scrum', 'rest api', 'graphql',
            'data analysis', 'data science', 'statistics', 'big data', 'hadoop', 'spark',
            'html', 'css', 'bootstrap', 'tailwind', 'typescript',
            'communication', 'leadership', 'teamwork', 'problem-solving',
            'project management', 'time management'
        ]
        
        self.education_levels = {
            'phd': 5,
            'doctoral': 5,
            'doctorate': 5,
            'master': 4,
            'mba': 4,
            'bachelor': 3,
            'bsc': 3,
            'b.tech': 3,
            'b.e': 3,
            'associate': 2,
            'diploma': 2,
            'high school': 1
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF resume"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        for skill in self.skills_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        return found_skills
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from resume"""
        # Pattern to match years of experience
        patterns = [
            r'(\d+)\+?\s*years?\s*(of\s*)?experience',
            r'experience\s*:\s*(\d+)\+?\s*years?',
            r'(\d+)\s*-\s*\d+\s*years?',
            r'(\d+)\s*years?\s*(of\s*)?relevant\s*experience'
        ]
        
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                years = [int(m) if isinstance(m, str) else int(m[0]) if m else 0 for m in matches]
                max_years = max(max_years, max(years))
        
        return max_years
    
    def extract_education_level(self, text: str) -> int:
        """Extract highest education level"""
        text_lower = text.lower()
        max_level = 0
        
        for edu, level in self.education_levels.items():
            if edu in text_lower:
                max_level = max(max_level, level)
        
        return max_level
    
    def extract_features(self, pdf_path: str) -> Dict:
        """Extract all features from resume"""
        text = self.extract_text_from_pdf(pdf_path)
        
        skills = self.extract_skills(text)
        years_exp = self.extract_experience_years(text)
        edu_level = self.extract_education_level(text)
        
        # Calculate skill score (normalized)
        skill_score = len(skills) / len(self.skills_keywords)
        
        return {
            'skills': skills,
            'skills_count': len(skills),
            'skill_score': skill_score,
            'experience_years': years_exp,
            'education_level': edu_level,
            'raw_text': text
        }


class FeatureEngineer:
    """Engineers features for the ML model"""
    
    def __init__(self):
        self.job_requirements_weights = {
            'technical_skills': 0.35,
            'experience': 0.30,
            'education': 0.20,
            'adaptability': 0.15
        }
    
    def calculate_qualification_score(
        self,
        skill_score: float,
        experience_years: int,
        education_level: int,
        job_requirements: Dict
    ) -> float:
        """Calculate qualification score based on features and job requirements"""
        
        # Normalize experience (cap at 15 years)
        exp_normalized = min(experience_years, 15) / 15
        
        # Normalize education (scale 0-5)
        edu_normalized = education_level / 5
        
        # Calculate weighted score
        technical_score = (
            skill_score * self.job_requirements_weights['technical_skills'] +
            exp_normalized * self.job_requirements_weights['experience'] +
            edu_normalized * self.job_requirements_weights['education']
        )
        
        return technical_score
    
    def prepare_features_for_model(
        self,
        skill_score: float,
        experience_years: int,
        education_level: int,
        adaptability_score: float = 0.5
    ) -> np.ndarray:
        """Prepare feature array for ML model prediction"""
        
        features = np.array([
            skill_score,
            min(experience_years, 15) / 15,
            education_level / 5,
            adaptability_score
        ])
        
        return features.reshape(1, -1)


class AdaptabilityEvaluator:
    """Evaluates applicant adaptability through structured questions"""
    
    def __init__(self):
        self.adaptability_questions = [
            {
                'id': 1,
                'question': 'Describe a time when you had to learn a new technology or skill quickly. How did you approach it?',
                'aspect': 'learning_speed'
            },
            {
                'id': 2,
                'question': 'Tell me about a situation where you had to adapt to significant changes in your work environment.',
                'aspect': 'adaptability'
            },
            {
                'id': 3,
                'question': 'How do you handle situations where you don\'t know the answer or need to solve an unfamiliar problem?',
                'aspect': 'problem_solving'
            },
            {
                'id': 4,
                'question': 'Describe a time when you had to work with a new team or in a new role. How did you adjust?',
                'aspect': 'team_adaptation'
            }
        ]
    
    def get_questions_for_position(self, position: str) -> List[Dict]:
        """Return tailored questions based on job position"""
        return self.adaptability_questions
    
    def calculate_adaptability_score(self, responses: List[str]) -> float:
        """
        Calculate adaptability score based on responses.
        In a production system, this would use NLP or ML for evaluation.
        Here we use a simple keyword-based scoring.
        """
        if not responses:
            return 0.5
        
        score = 0
        positive_keywords = [
            'learned', 'adapted', 'developed', 'improved', 'innovated',
            'collaborated', 'problem', 'solution', 'challenge', 'success',
            'team', 'quickly', 'effectively', 'efficiently', 'managed'
        ]
        
        for response in responses:
            response_lower = response.lower()
            keyword_count = sum(1 for kw in positive_keywords if kw in response_lower)
            score += min(keyword_count / 5, 1.0)  # Cap at 1 per response
        
        return min(score / len(responses), 1.0)

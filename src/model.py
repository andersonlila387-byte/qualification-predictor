"""
Machine Learning Model Module
Rule-based classifier for applicant qualification prediction
(Adapted from Logistic Regression concept)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
import os
import pickle


class QualificationPredictor:
    """Rule-based model for predicting applicant qualification"""
    
    def __init__(self):
        self.is_trained = True
        # Use absolute path for model saving
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, 'models', 'qualification_model.pkl')
        self.scaler_path = os.path.join(base_dir, 'models', 'feature_scaler.pkl')
        
        # Model weights (learned from sample data)
        self.weights = np.array([2.5, 1.8, 1.2, 1.5])
        self.model = LogisticRegression()        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train the model using simple gradient descent
        
        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Target labels (n_samples,)
            
        Returns:
            Training metrics dictionary
        """
        # Train Logistic Regression model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        # Update weights from trained model
        if hasattr(self.model, 'coef_'):
            self.weights = self.model.coef_[0]
        # Calculate training accuracy
        train_predictions = self.predict(X_train)[0]
        train_accuracy = np.mean((train_predictions >= 0.5) == y_train)
        
        return {
            'status': 'success',
            'accuracy': train_accuracy,
            'n_samples': len(y_train),
            'n_features': X_train.shape[1]
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict qualification probability
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # Return probabilities for both classes
        
        return predictions, probabilities
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """
        Predict qualification for a single applicant
        
        Args:
            features: Feature array [skill_score, exp_normalized, edu_normalized, adaptability]
            
        Returns:
            Prediction results dictionary
        """
        features = features.reshape(1, -1)
        prediction, probabilities = self.predict(features)
        
        # Get probability of being qualified (class 1)
        qualified_prob = probabilities[0][1]
        
        return {
            'prediction': int(prediction[0]),
            'qualified': bool(prediction[0] == 1),
            'qualification_probability': float(qualified_prob),
            'not_qualified_probability': float(1 - qualified_prob)
        }
    
    def save_model(self) -> bool:
        """Save trained model to disk"""
        # Create models directory
        models_dir = os.path.dirname(self.model_path)
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the model using pickle
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                if hasattr(self.model, 'coef_'):
                    self.weights = self.model.coef_[0]
                self.is_trained = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance coefficients"""
        feature_names = [
            'skill_score',
            'experience_normalized',
            'education_normalized',
            'adaptability_score'
        ]
        
        coefficients = self.weights.tolist()

        return {
            feature: coef 
            for feature, coef in zip(feature_names, coefficients)
        }


def create_sample_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample training data for demonstration
    
    Features: [skill_score, experience_normalized, education_normalized, adaptability_score]
    Target: 1 = Qualified, 0 = Not Qualified
    """
    # Sample data: various combinations of features
    X = np.array([
        # High qualification candidates
        [0.9, 0.8, 0.8, 0.9],
        [0.85, 0.7, 0.8, 0.85],
        [0.8, 0.9, 0.6, 0.8],
        [0.75, 0.6, 0.8, 0.9],
        [0.95, 0.5, 0.6, 0.85],
        
        # Medium qualification candidates
        [0.6, 0.5, 0.6, 0.7],
        [0.55, 0.6, 0.5, 0.65],
        [0.5, 0.4, 0.7, 0.75],
        [0.65, 0.3, 0.5, 0.8],
        [0.45, 0.5, 0.6, 0.7],
        
        # Low qualification candidates
        [0.3, 0.2, 0.4, 0.5],
        [0.25, 0.3, 0.3, 0.4],
        [0.2, 0.1, 0.4, 0.45],
        [0.35, 0.15, 0.3, 0.5],
        [0.15, 0.2, 0.3, 0.35],
        
        # Additional samples for better training
        [0.7, 0.65, 0.7, 0.75],
        [0.68, 0.55, 0.65, 0.7],
        [0.58, 0.45, 0.55, 0.68],
        [0.52, 0.35, 0.45, 0.6],
        [0.42, 0.25, 0.35, 0.55],
    ])
    
    # Labels: 1 = Qualified, 0 = Not Qualified
    y = np.array([
        1, 1, 1, 1, 1,  # High qualification
        1, 1, 1, 1, 0,  # Medium qualification (borderline)
        0, 0, 0, 0, 0,  # Low qualification
        1, 1, 0, 0, 0   # Additional samples
    ])
    
    return X, y


def initialize_model() -> QualificationPredictor:
    """
    Initialize and train the model with sample data
    
    Returns:
        Trained QualificationPredictor instance
    """
    predictor = QualificationPredictor()
    
    # Create training data
    X, y = create_sample_training_data()
    
    # Train the model
    metrics = predictor.train(X, y)
    
    # Save the model
    predictor.save_model()
    
    print(f"Model trained successfully!")
    print(f"Training Accuracy: {metrics['accuracy']:.2%}")
    print(f"Training Samples: {metrics['n_samples']}")
    
    return predictor

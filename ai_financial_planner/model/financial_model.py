"""
AI Financial Planning Model
This module contains the machine learning model for personalized financial planning recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialPlannerModel:
    """
    A comprehensive financial planning model that provides personalized investment recommendations
    based on user financial data.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'recommended_plan'
        self.model_type = 'RandomForest'
        
    def load_data(self, data_path='data/user_data.csv'):
        """Load and preprocess the financial dataset."""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: Dataset not found at {data_path}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for model training."""
        # Create a copy to avoid modifying original data
        df = self.data.copy()
        
        # Handle categorical variables
        categorical_columns = ['risk_tolerance', 'investment_experience']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Define feature columns (exclude target and non-predictive columns)
        self.feature_columns = [col for col in df.columns 
                               if col not in [self.target_column, 'age']]  # Age might be less predictive
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data preprocessed. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
    def train_model(self, model_type='RandomForest'):
        """Train the financial planning model."""
        self.model_type = model_type
        
        if model_type == 'RandomForest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'GradientBoosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'LinearRegression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        if model_type == 'LinearRegression':
            self.model.fit(self.X_train_scaled, self.y_train)
        else:
            self.model.fit(self.X_train, self.y_train)
        
        print(f"{model_type} model trained successfully!")
        
    def evaluate_model(self):
        """Evaluate the model performance."""
        if self.model is None:
            print("No model trained yet!")
            return
        
        # Make predictions
        if self.model_type == 'LinearRegression':
            y_pred = self.model.predict(self.X_test_scaled)
        else:
            y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\n=== Model Evaluation ===")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        if self.model is None:
            print("No model trained yet!")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n=== Feature Importance ===")
            print(importance_df)
            return importance_df
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def predict(self, user_data):
        """
        Make a prediction for a single user.
        
        Args:
            user_data (dict): Dictionary containing user financial information
                Expected keys: income, expenses, savings, goals, debt_ratio, 
                risk_tolerance, investment_experience
        
        Returns:
            dict: Prediction results including recommended_plan and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in user_df.columns:
                user_df[col] = encoder.transform(user_df[col])
        
        # Select features
        user_features = user_df[self.feature_columns]
        
        # Make prediction
        if self.model_type == 'LinearRegression':
            prediction = self.model.predict(self.scaler.transform(user_features))[0]
        else:
            prediction = self.model.predict(user_features)[0]
        
        # Calculate confidence based on feature similarity to training data
        confidence = self._calculate_confidence(user_features)
        
        return {
            'recommended_plan': max(0, prediction),  # Ensure non-negative
            'confidence': confidence,
            'model_type': self.model_type
        }
    
    def _calculate_confidence(self, user_features):
        """Calculate prediction confidence based on feature similarity."""
        # If training data is not available, return a default confidence
        if not hasattr(self, 'X_train') or self.X_train is None:
            return 0.8  # Default confidence when training data not available
        
        # Simple confidence calculation based on how close user features are to training data
        distances = np.linalg.norm(self.X_train - user_features.iloc[0], axis=1)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Normalize to 0-1 scale (closer to training data = higher confidence)
        confidence = 1 - (min_distance / (max_distance + 1e-8))
        return min(1.0, max(0.0, confidence))
    
    def save_model(self, model_path='model/financial_model.pkl'):
        """Save the trained model and preprocessors."""
        if self.model is None:
            print("No model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='model/financial_model.pkl'):
        """Load a pre-trained model."""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            return False
    
    def plot_feature_importance(self, save_path='static/feature_importance.png'):
        """Plot feature importance."""
        importance_df = self.get_feature_importance()
        if importance_df is None:
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance in Financial Planning Model')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to {save_path}")

def train_and_save_model():
    """Convenience function to train and save the model."""
    # Initialize model
    model = FinancialPlannerModel()
    
    # Load and preprocess data
    if not model.load_data():
        return None
    
    model.preprocess_data()
    
    # Train different models and compare
    models_to_try = ['RandomForest', 'GradientBoosting', 'LinearRegression']
    best_model = None
    best_r2 = -np.inf
    
    for model_type in models_to_try:
        print(f"\n=== Training {model_type} ===")
        model.train_model(model_type)
        metrics = model.evaluate_model()
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = model_type
    
    print(f"\nBest model: {best_model} with R² = {best_r2:.4f}")
    
    # Retrain with best model
    model.train_model(best_model)
    model.evaluate_model()
    model.get_feature_importance()
    
    # Save the model
    model.save_model()
    
    return model

if __name__ == "__main__":
    # Train and save the model
    model = train_and_save_model()
    
    if model:
        # Example prediction
        example_user = {
            'income': 75000,
            'expenses': 50000,
            'savings': 15000,
            'goals': 200000,
            'debt_ratio': 0.1,
            'risk_tolerance': 'medium',
            'investment_experience': 'intermediate'
        }
        
        prediction = model.predict(example_user)
        print(f"\nExample Prediction:")
        print(f"Recommended Investment Plan: ${prediction['recommended_plan']:.2f}")
        print(f"Confidence: {prediction['confidence']:.2%}")

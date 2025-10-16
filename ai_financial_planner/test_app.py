#!/usr/bin/env python3
"""
Test script for AI Financial Planning Assistant
This script demonstrates the core functionality of the application.
"""

import sys
import os
import json
from datetime import datetime

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.financial_model import FinancialPlannerModel

def test_model_prediction():
    """Test the ML model prediction functionality."""
    print("🤖 Testing AI Financial Planning Model")
    print("=" * 50)
    
    # Load the model
    model = FinancialPlannerModel()
    if not model.load_model():
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully!")
    print(f"Model type: {model.model_type}")
    print(f"Features: {model.feature_columns}")
    
    # Test cases for Indian market (in Lakhs INR)
    test_cases = [
        {
            "name": "Young Professional (India)",
            "data": {
                "income": 6.0,  # 6L INR
                "expenses": 4.5,  # 4.5L INR
                "savings": 1.0,  # 1L INR
                "goals": 15.0,  # 15L INR
                "debt_ratio": 0.15,
                "risk_tolerance": "medium",
                "investment_experience": "beginner"
            }
        },
        {
            "name": "Mid-Career Professional (India)",
            "data": {
                "income": 12.0,  # 12L INR
                "expenses": 8.0,  # 8L INR
                "savings": 3.0,  # 3L INR
                "goals": 50.0,  # 50L INR
                "debt_ratio": 0.1,
                "risk_tolerance": "medium",
                "investment_experience": "intermediate"
            }
        },
        {
            "name": "High Earner (India)",
            "data": {
                "income": 25.0,  # 25L INR
                "expenses": 15.0,  # 15L INR
                "savings": 8.0,  # 8L INR
                "goals": 100.0,  # 100L INR
                "debt_ratio": 0.05,
                "risk_tolerance": "high",
                "investment_experience": "advanced"
            }
        }
    ]
    
    print("\n📊 Running Test Cases:")
    print("-" * 30)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Income: ₹{test_case['data']['income']:.1f}L")
        print(f"   Expenses: ₹{test_case['data']['expenses']:.1f}L")
        print(f"   Savings: ₹{test_case['data']['savings']:.1f}L")
        print(f"   Goals: ₹{test_case['data']['goals']:.1f}L")
        
        try:
            # Make prediction
            prediction = model.predict(test_case['data'])
            
            print(f"   💡 Recommended Investment: ₹{prediction['recommended_plan']:.2f}L")
            print(f"   🎯 Confidence: {prediction['confidence']:.1%}")
            
            # Calculate some basic insights
            income = test_case['data']['income']
            expenses = test_case['data']['expenses']
            savings = test_case['data']['savings']
            goals = test_case['data']['goals']
            recommended = prediction['recommended_plan']
            
            savings_rate = (savings / income) * 100
            expense_rate = (expenses / income) * 100
            goal_timeline = goals / recommended if recommended > 0 else float('inf')
            
            print(f"   📈 Savings Rate: {savings_rate:.1f}%")
            print(f"   💸 Expense Rate: {expense_rate:.1f}%")
            print(f"   ⏰ Goal Timeline: {goal_timeline:.1f} years")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False
    
    return True

def test_api_simulation():
    """Simulate API calls without running the Flask server."""
    print("\n🌐 Testing API Simulation")
    print("=" * 50)
    
    # Simulate the API logic
    model = FinancialPlannerModel()
    if not model.load_model():
        print("❌ Failed to load model for API simulation")
        return False
    
    # Test API request data
    api_request = {
        "income": 75000,
        "expenses": 50000,
        "savings": 15000,
        "goals": 200000,
        "debt_ratio": 0.1,
        "risk_tolerance": "medium",
        "investment_experience": "intermediate"
    }
    
    print("📤 API Request:")
    print(json.dumps(api_request, indent=2))
    
    try:
        # Simulate prediction
        prediction = model.predict(api_request)
        
        # Generate insights (simplified version)
        income = api_request['income']
        expenses = api_request['expenses']
        savings = api_request['savings']
        goals = api_request['goals']
        recommended = prediction['recommended_plan']
        
        insights = {
            "savings_rate": round((savings / income) * 100, 2),
            "expense_ratio": round((expenses / income) * 100, 2),
            "goal_achievement_time_years": round(goals / recommended, 1) if recommended > 0 else float('inf'),
            "recommendations": []
        }
        
        # Add recommendations
        if insights['savings_rate'] < 10:
            insights['recommendations'].append("Consider increasing your savings rate. Aim for at least 20% of your income.")
        elif insights['savings_rate'] > 30:
            insights['recommendations'].append("Great job! You have a healthy savings rate. Consider investing more aggressively.")
        
        if insights['expense_ratio'] > 80:
            insights['recommendations'].append("Your expenses are quite high relative to income. Look for ways to reduce costs.")
        
        if insights['goal_achievement_time_years'] > 20:
            insights['recommendations'].append("Your goals may take a long time to achieve. Consider increasing your investment amount or adjusting your goals.")
        
        if not insights['recommendations']:
            insights['recommendations'].append("Your financial situation looks balanced. Keep up the good work!")
        
        # Simulate API response
        api_response = {
            "success": True,
            "prediction": prediction,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n📥 API Response:")
        print(json.dumps(api_response, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ API simulation error: {str(e)}")
        return False

def test_data_analysis():
    """Test data analysis functionality."""
    print("\n📊 Testing Data Analysis")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load the dataset
        data = pd.read_csv('data/user_data.csv')
        print(f"✅ Dataset loaded: {data.shape[0]} records, {data.shape[1]} features")
        
        # Basic statistics
        print("\n📈 Dataset Statistics:")
        print(f"   Average Income: ${data['income'].mean():,.2f}")
        print(f"   Average Expenses: ${data['expenses'].mean():,.2f}")
        print(f"   Average Savings: ${data['savings'].mean():,.2f}")
        print(f"   Average Goals: ${data['goals'].mean():,.2f}")
        
        # Risk tolerance distribution
        print("\n🎯 Risk Tolerance Distribution:")
        risk_dist = data['risk_tolerance'].value_counts()
        for risk, count in risk_dist.items():
            print(f"   {risk.capitalize()}: {count} ({count/len(data)*100:.1f}%)")
        
        # Investment experience distribution
        print("\n💼 Investment Experience Distribution:")
        exp_dist = data['investment_experience'].value_counts()
        for exp, count in exp_dist.items():
            print(f"   {exp.capitalize()}: {count} ({count/len(data)*100:.1f}%)")
        
        # Correlation analysis
        print("\n🔗 Feature Correlations:")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        # Show correlations with recommended_plan
        if 'recommended_plan' in corr_matrix.columns:
            plan_corr = corr_matrix['recommended_plan'].sort_values(ascending=False)
            for feature, corr in plan_corr.items():
                if feature != 'recommended_plan':
                    print(f"   {feature}: {corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data analysis error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Financial Planning Assistant - Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Model Prediction", test_model_prediction),
        ("API Simulation", test_api_simulation),
        ("Data Analysis", test_data_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} test passed!")
            else:
                print(f"❌ {test_name} test failed!")
        except Exception as e:
            print(f"❌ {test_name} test error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print("-" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The application is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

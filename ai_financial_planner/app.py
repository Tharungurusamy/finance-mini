"""
AI Financial Planning Assistant - Flask API
This module provides REST API endpoints for the financial planning application.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import json
from datetime import datetime

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

try:
    from financial_model import FinancialPlannerModel
    from ai_chatbot import chatbot
except ImportError:
    # If import fails, try adding current directory to path
    sys.path.append(os.path.dirname(__file__))
    from model.financial_model import FinancialPlannerModel
    from ai_chatbot import chatbot

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the model
model = FinancialPlannerModel()

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Serve the comprehensive financial dashboard."""
    return render_template('dashboard.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model.model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict financial planning recommendations based on user data.
    
    Expected JSON payload:
    {
        "income": float,
        "expenses": float,
        "savings": float,
        "goals": float,
        "debt_ratio": float (optional, default 0.1),
        "risk_tolerance": str (optional, "low"/"medium"/"high", default "medium"),
        "investment_experience": str (optional, "beginner"/"intermediate"/"advanced", default "intermediate")
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['income', 'expenses', 'savings', 'goals']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Set default values for optional fields
        user_data = {
            'income': float(data['income']),
            'expenses': float(data['expenses']),
            'savings': float(data['savings']),
            'goals': float(data['goals']),
            'debt_ratio': float(data.get('debt_ratio', 0.1)),
            'risk_tolerance': data.get('risk_tolerance', 'medium'),
            'investment_experience': data.get('investment_experience', 'intermediate')
        }
        
        # Validate data ranges
        if user_data['income'] <= 0:
            return jsonify({'error': 'Income must be positive'}), 400
        
        if user_data['expenses'] < 0:
            return jsonify({'error': 'Expenses cannot be negative'}), 400
        
        if user_data['savings'] < 0:
            return jsonify({'error': 'Savings cannot be negative'}), 400
        
        if user_data['goals'] <= 0:
            return jsonify({'error': 'Goals must be positive'}), 400
        
        if not 0 <= user_data['debt_ratio'] <= 1:
            return jsonify({'error': 'Debt ratio must be between 0 and 1'}), 400
        
        # Validate categorical variables
        valid_risk_levels = ['low', 'medium', 'high']
        if user_data['risk_tolerance'] not in valid_risk_levels:
            return jsonify({
                'error': f'Risk tolerance must be one of: {", ".join(valid_risk_levels)}'
            }), 400
        
        valid_experience_levels = ['beginner', 'intermediate', 'advanced']
        if user_data['investment_experience'] not in valid_experience_levels:
            return jsonify({
                'error': f'Investment experience must be one of: {", ".join(valid_experience_levels)}'
            }), 400
        
        # Make prediction
        prediction = model.predict(user_data)
        
        # Add additional insights
        insights = generate_insights(user_data, prediction)
        
        response = {
            'success': True,
            'prediction': prediction,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/insights', methods=['POST'])
def get_insights():
    """Get detailed financial insights and recommendations."""
    try:
        data = request.get_json()
        user_data = data.get('user_data', {})
        
        insights = generate_detailed_insights(user_data)
        
        return jsonify({
            'success': True,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Insights generation failed: {str(e)}'}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if model.model is None:
        return jsonify({
            'model_loaded': False,
            'message': 'No model loaded'
        })
    
    return jsonify({
        'model_loaded': True,
        'model_type': model.model_type,
        'feature_columns': model.feature_columns,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    AI Chatbot endpoint for answering financial questions and clarifying doubts.
    
    Expected JSON payload:
    {
        "question": "What should I invest in?",
        "user_data": {
            "income": 8.5,
            "expenses": 6.0,
            "savings": 2.0,
            "goals": 50.0,
            "risk_tolerance": "medium",
            "investment_experience": "intermediate"
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        user_data = data.get('user_data', {})
        
        # Get AI response
        response = chatbot.analyze_question(question, user_data)
        
        # Add personalized advice if user data is available
        if user_data:
            personalized_advice = chatbot.get_personalized_advice(user_data, response['type'])
            response['personalized_advice'] = personalized_advice
        
        # Add timestamp
        response['timestamp'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/api/portfolio/analysis', methods=['POST'])
def portfolio_analysis():
    """Get comprehensive portfolio analysis and recommendations."""
    try:
        data = request.get_json()
        user_data = data.get('user_data', {})
        
        # Calculate portfolio metrics
        income = float(user_data.get('income', 0))
        expenses = float(user_data.get('expenses', 0))
        savings = float(user_data.get('savings', 0))
        goals = float(user_data.get('goals', 0))
        age = int(user_data.get('age', 30))
        risk_tolerance = user_data.get('risk_tolerance', 'medium')
        
        # Calculate financial health metrics
        savings_rate = (savings / income * 100) if income > 0 else 0
        expense_ratio = (expenses / income * 100) if income > 0 else 0
        debt_ratio = float(user_data.get('debt_ratio', 0)) * 100
        
        # Calculate emergency fund months
        emergency_fund_months = (savings * 12) / expenses if expenses > 0 else 0
        
        # Asset allocation recommendation based on age and risk tolerance
        if age < 30:
            equity_allocation = 80 if risk_tolerance == 'high' else 70
        elif age < 50:
            equity_allocation = 70 if risk_tolerance == 'high' else 60
        else:
            equity_allocation = 60 if risk_tolerance == 'high' else 50
            
        debt_allocation = 100 - equity_allocation - 10  # 10% for gold/others
        gold_allocation = 10
        
        # Generate recommendations
        recommendations = []
        if savings_rate < 20:
            recommendations.append("Increase your savings rate to at least 20% of income")
        if emergency_fund_months < 6:
            recommendations.append("Build emergency fund to cover 6 months of expenses")
        if debt_ratio > 30:
            recommendations.append("Consider reducing debt before increasing investments")
        if equity_allocation > 80:
            recommendations.append("Consider reducing equity allocation for better risk management")
            
        # Goal analysis
        goal_analysis = []
        if goals > 0:
            years_to_goal = goals / (savings * 12) if savings > 0 else 999
            if years_to_goal > 20:
                goal_analysis.append({
                    'goal': 'Primary Goal',
                    'target': goals,
                    'current': savings * 12,
                    'timeline_years': years_to_goal,
                    'recommendation': 'Consider increasing monthly savings or adjusting goal timeline'
                })
        
        return jsonify({
            'success': True,
            'portfolio_metrics': {
                'total_value': savings * 12,
                'savings_rate': savings_rate,
                'expense_ratio': expense_ratio,
                'debt_ratio': debt_ratio,
                'emergency_fund_months': emergency_fund_months
            },
            'asset_allocation': {
                'equity': equity_allocation,
                'debt': debt_allocation,
                'gold': gold_allocation
            },
            'recommendations': recommendations,
            'goal_analysis': goal_analysis,
            'risk_analysis': {
                'overall_risk': risk_tolerance,
                'volatility': 'moderate' if risk_tolerance == 'medium' else risk_tolerance,
                'diversification_score': 8 if equity_allocation <= 70 else 6,
                'liquidity_score': 9 if emergency_fund_months >= 6 else 5
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Portfolio analysis failed: {str(e)}'}), 500

@app.route('/api/chat/test', methods=['POST'])
def test_chat():
    """Simple test endpoint for debugging."""
    try:
        data = request.get_json()
        print(f"Test endpoint received: {data}")
        return jsonify({'success': True, 'received': data})
    except Exception as e:
        print(f"Test endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/suggestions', methods=['GET'])
def get_chat_suggestions():
    """Get suggested questions for the chatbot."""
    suggestions = [
        "What should I invest in?",
        "How much should I save?",
        "Explain my financial plan",
        "What are good investment options in India?",
        "How to build an emergency fund?",
        "Best tax saving investments",
        "PPF vs ELSS comparison",
        "How to increase my savings?",
        "Retirement planning in India",
        "What's my risk tolerance?"
    ]
    
    return jsonify({
        'success': True,
        'suggestions': suggestions,
        'timestamp': datetime.now().isoformat()
    })

def generate_insights(user_data, prediction):
    """Generate basic financial insights."""
    income = user_data['income']
    expenses = user_data['expenses']
    savings = user_data['savings']
    goals = user_data['goals']
    recommended_plan = prediction['recommended_plan']
    
    # Calculate key ratios
    savings_rate = (savings / income) * 100 if income > 0 else 0
    expense_ratio = (expenses / income) * 100 if income > 0 else 0
    goal_achievement_time = goals / recommended_plan if recommended_plan > 0 else float('inf')
    
    insights = {
        'savings_rate': round(savings_rate, 2),
        'expense_ratio': round(expense_ratio, 2),
        'goal_achievement_time_years': round(goal_achievement_time, 1),
        'recommendations': []
    }
    
    # Generate recommendations based on financial health
    if savings_rate < 10:
        insights['recommendations'].append(
            "Consider increasing your savings rate. Aim for at least 20% of your income."
        )
    elif savings_rate > 30:
        insights['recommendations'].append(
            "Great job! You have a healthy savings rate. Consider investing more aggressively."
        )
    
    if expense_ratio > 80:
        insights['recommendations'].append(
            "Your expenses are quite high relative to income. Look for ways to reduce costs."
        )
    
    if goal_achievement_time > 20:
        insights['recommendations'].append(
            "Your goals may take a long time to achieve. Consider increasing your investment amount or adjusting your goals."
        )
    
    if not insights['recommendations']:
        insights['recommendations'].append(
            "Your financial situation looks balanced. Keep up the good work!"
        )
    
    return insights

def generate_detailed_insights(user_data):
    """Generate detailed financial insights and recommendations."""
    income = user_data.get('income', 0)
    expenses = user_data.get('expenses', 0)
    savings = user_data.get('savings', 0)
    goals = user_data.get('goals', 0)
    debt_ratio = user_data.get('debt_ratio', 0)
    risk_tolerance = user_data.get('risk_tolerance', 'medium')
    
    insights = {
        'financial_health_score': calculate_financial_health_score(user_data),
        'budget_analysis': analyze_budget(income, expenses, savings),
        'investment_strategy': get_investment_strategy(risk_tolerance, user_data),
        'goal_analysis': analyze_goals(goals, income, savings),
        'debt_analysis': analyze_debt(debt_ratio, income),
        'next_steps': get_next_steps(user_data)
    }
    
    return insights

def calculate_financial_health_score(user_data):
    """Calculate a financial health score from 0-100."""
    income = user_data.get('income', 0)
    expenses = user_data.get('expenses', 0)
    savings = user_data.get('savings', 0)
    debt_ratio = user_data.get('debt_ratio', 0)
    
    score = 0
    
    # Savings rate (40% of score)
    if income > 0:
        savings_rate = savings / income
        if savings_rate >= 0.2:
            score += 40
        elif savings_rate >= 0.1:
            score += 30
        elif savings_rate >= 0.05:
            score += 20
        else:
            score += 10
    
    # Expense control (30% of score)
    if income > 0:
        expense_ratio = expenses / income
        if expense_ratio <= 0.5:
            score += 30
        elif expense_ratio <= 0.7:
            score += 25
        elif expense_ratio <= 0.8:
            score += 15
        else:
            score += 5
    
    # Debt management (30% of score)
    if debt_ratio <= 0.1:
        score += 30
    elif debt_ratio <= 0.2:
        score += 25
    elif debt_ratio <= 0.3:
        score += 15
    else:
        score += 5
    
    return min(100, max(0, score))

def analyze_budget(income, expenses, savings):
    """Analyze the user's budget."""
    if income <= 0:
        return {"status": "error", "message": "Invalid income data"}
    
    expense_ratio = (expenses / income) * 100
    savings_ratio = (savings / income) * 100
    
    return {
        "monthly_income": income / 12,
        "monthly_expenses": expenses / 12,
        "monthly_savings": savings / 12,
        "expense_ratio": round(expense_ratio, 2),
        "savings_ratio": round(savings_ratio, 2),
        "status": "healthy" if expense_ratio <= 70 and savings_ratio >= 20 else "needs_improvement"
    }

def get_investment_strategy(risk_tolerance, user_data):
    """Get investment strategy recommendations."""
    strategies = {
        'low': {
            'allocation': {'Bonds': 70, 'Stocks': 20, 'Cash': 10},
            'description': 'Conservative approach focusing on capital preservation'
        },
        'medium': {
            'allocation': {'Bonds': 40, 'Stocks': 50, 'Cash': 10},
            'description': 'Balanced approach with moderate risk and return'
        },
        'high': {
            'allocation': {'Bonds': 20, 'Stocks': 70, 'Cash': 10},
            'description': 'Aggressive approach focusing on growth'
        }
    }
    
    return strategies.get(risk_tolerance, strategies['medium'])

def analyze_goals(goals, income, savings):
    """Analyze financial goals feasibility."""
    if income <= 0 or goals <= 0:
        return {"status": "error", "message": "Invalid goal or income data"}
    
    goal_ratio = goals / income
    years_to_achieve = goals / savings if savings > 0 else float('inf')
    
    return {
        "goal_amount": goals,
        "goal_to_income_ratio": round(goal_ratio, 2),
        "years_to_achieve": round(years_to_achieve, 1),
        "feasibility": "achievable" if years_to_achieve <= 20 else "challenging"
    }

def analyze_debt(debt_ratio, income):
    """Analyze debt situation."""
    debt_amount = debt_ratio * income
    
    return {
        "debt_ratio": round(debt_ratio * 100, 2),
        "debt_amount": round(debt_amount, 2),
        "status": "healthy" if debt_ratio <= 0.2 else "needs_attention"
    }

def get_next_steps(user_data):
    """Get actionable next steps for the user."""
    steps = []
    
    income = user_data.get('income', 0)
    expenses = user_data.get('expenses', 0)
    savings = user_data.get('savings', 0)
    debt_ratio = user_data.get('debt_ratio', 0)
    
    if income > 0:
        savings_rate = savings / income
        expense_rate = expenses / income
        
        if savings_rate < 0.1:
            steps.append("Start an emergency fund with 3-6 months of expenses")
            steps.append("Create a budget to track and reduce expenses")
        
        if expense_rate > 0.8:
            steps.append("Review and cut unnecessary expenses")
            steps.append("Consider increasing your income through side hustles")
        
        if debt_ratio > 0.2:
            steps.append("Focus on paying down high-interest debt")
            steps.append("Consider debt consolidation if beneficial")
        
        if savings_rate >= 0.2 and debt_ratio <= 0.1:
            steps.append("Start investing in diversified portfolios")
            steps.append("Consider tax-advantaged accounts (401k, IRA)")
    
    if not steps:
        steps.append("Continue monitoring your financial health")
        steps.append("Review and adjust your financial plan quarterly")
    
    return steps

if __name__ == '__main__':
    # Try to load existing model, if not available, train a new one
    if not model.load_model():
        print("No pre-trained model found. Training a new model...")
        try:
            from financial_model import train_and_save_model
        except ImportError:
            from model.financial_model import train_and_save_model
        model = train_and_save_model()
    
    if model and model.model is not None:
        print("Model loaded successfully!")
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load or train model. Please check the data and try again.")

"""
AI Financial Planning Assistant - Streamlit Dashboard
This module provides an interactive web interface for the financial planning application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import requests
import json
from datetime import datetime

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from financial_model import FinancialPlannerModel

# Page configuration
st.set_page_config(
    page_title="AI Financial Planning Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample financial data for demonstration."""
    try:
        data = pd.read_csv('data/user_data.csv')
        return data
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    """Load the financial planning model."""
    model = FinancialPlannerModel()
    if model.load_model():
        return model
    else:
        st.error("Failed to load the financial planning model. Please ensure the model is trained.")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Financial Planning Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Financial Planning", "Data Analysis", "Model Information", "About"]
    )
    
    if page == "Financial Planning":
        financial_planning_page()
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "Model Information":
        model_info_page()
    elif page == "About":
        about_page()

def financial_planning_page():
    """Financial planning input and prediction page."""
    st.header("💰 Get Your Personalized Financial Plan")
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Financial Information")
        
        # Financial inputs
        income = st.number_input(
            "Annual Income ($)",
            min_value=0,
            value=75000,
            step=1000,
            help="Your total annual income before taxes"
        )
        
        expenses = st.number_input(
            "Annual Expenses ($)",
            min_value=0,
            value=50000,
            step=1000,
            help="Your total annual expenses"
        )
        
        savings = st.number_input(
            "Annual Savings ($)",
            min_value=0,
            value=15000,
            step=1000,
            help="Amount you save annually"
        )
        
        goals = st.number_input(
            "Financial Goals ($)",
            min_value=0,
            value=200000,
            step=1000,
            help="Your target financial goal amount"
        )
        
        st.subheader("Additional Information")
        
        debt_ratio = st.slider(
            "Debt Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Debt as a percentage of your income"
        )
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["low", "medium", "high"],
            index=1,
            help="Your comfort level with investment risk"
        )
        
        investment_experience = st.selectbox(
            "Investment Experience",
            ["beginner", "intermediate", "advanced"],
            index=1,
            help="Your level of investment experience"
        )
        
        # Predict button
        if st.button("🚀 Get AI Recommendations", type="primary", use_container_width=True):
            make_prediction(income, expenses, savings, goals, debt_ratio, risk_tolerance, investment_experience)
    
    with col2:
        st.subheader("Financial Health Overview")
        
        if income > 0:
            # Calculate basic metrics
            savings_rate = (savings / income) * 100
            expense_rate = (expenses / income) * 100
            debt_amount = debt_ratio * income
            
            # Display metrics
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("Savings Rate", f"{savings_rate:.1f}%")
                st.metric("Expense Rate", f"{expense_rate:.1f}%")
            
            with col2_2:
                st.metric("Debt Amount", f"${debt_amount:,.0f}")
                st.metric("Net Worth", f"${savings - debt_amount:,.0f}")
            
            # Financial health gauge
            health_score = calculate_financial_health_score(savings_rate, expense_rate, debt_ratio)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Financial Health Score"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def make_prediction(income, expenses, savings, goals, debt_ratio, risk_tolerance, investment_experience):
    """Make prediction using the loaded model."""
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model not available. Please check the model files.")
        return
    
    # Prepare user data
    user_data = {
        'income': income,
        'expenses': expenses,
        'savings': savings,
        'goals': goals,
        'debt_ratio': debt_ratio,
        'risk_tolerance': risk_tolerance,
        'investment_experience': investment_experience
    }
    
    try:
        # Make prediction
        prediction = model.predict(user_data)
        
        # Display results
        st.success("✅ Prediction completed successfully!")
        
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <h2>Recommended Annual Investment</h2>
            <h1>${prediction['recommended_plan']:,.0f}</h1>
            <p>Confidence: {prediction['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate insights
        insights = generate_insights(user_data, prediction)
        
        # Display insights
        st.subheader("📊 Financial Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>Savings Analysis</h4>
                <p>Your savings rate is {insights['savings_rate']:.1f}% of your income.</p>
                <p>This is {'excellent' if insights['savings_rate'] > 20 else 'good' if insights['savings_rate'] > 10 else 'needs improvement'}.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <h4>Goal Timeline</h4>
                <p>At this investment rate, you'll reach your goal in approximately {insights['goal_achievement_time_years']:.1f} years.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(insights['recommendations'], 1):
            st.markdown(f"{i}. {rec}")
        
        # Investment strategy based on risk tolerance
        st.subheader("📈 Investment Strategy")
        strategy = get_investment_strategy(risk_tolerance)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recommended Allocation:**")
            for asset, percentage in strategy['allocation'].items():
                st.write(f"• {asset}: {percentage}%")
        
        with col2:
            st.write(f"**Strategy Description:**")
            st.write(strategy['description'])
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def data_analysis_page():
    """Data analysis and visualization page."""
    st.header("📊 Financial Data Analysis")
    
    # Load sample data
    data = load_sample_data()
    if data is None:
        st.error("Sample data not found. Please ensure the dataset is available.")
        return
    
    st.subheader("Dataset Overview")
    st.write(f"Dataset contains {len(data)} records with {len(data.columns)} features.")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(data.head(10))
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(data.describe())
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Distribution")
        fig = px.histogram(data, x='income', nbins=30, title="Distribution of Annual Income")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Savings vs Income")
        fig = px.scatter(data, x='income', y='savings', color='risk_tolerance',
                        title="Savings vs Income by Risk Tolerance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Correlation Matrix of Financial Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk tolerance analysis
    st.subheader("Risk Tolerance Analysis")
    risk_analysis = data.groupby('risk_tolerance').agg({
        'income': 'mean',
        'savings': 'mean',
        'recommended_plan': 'mean'
    }).round(2)
    
    st.dataframe(risk_analysis)
    
    # Investment experience analysis
    st.subheader("Investment Experience Analysis")
    exp_analysis = data.groupby('investment_experience').agg({
        'income': 'mean',
        'savings': 'mean',
        'recommended_plan': 'mean'
    }).round(2)
    
    st.dataframe(exp_analysis)

def model_info_page():
    """Model information and performance page."""
    st.header("🤖 Model Information")
    
    model = load_model()
    if model is None:
        st.error("Model not available.")
        return
    
    st.subheader("Model Details")
    st.write(f"**Model Type:** {model.model_type}")
    st.write(f"**Features Used:** {', '.join(model.feature_columns)}")
    
    # Feature importance
    if hasattr(model.model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': model.feature_columns,
            'Importance': model.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                    title="Feature Importance in Financial Planning Model")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(importance_df)
    
    # Model performance (if available)
    st.subheader("Model Performance")
    st.info("To see model performance metrics, please run the model training script.")

def about_page():
    """About page with application information."""
    st.header("About AI Financial Planning Assistant")
    
    st.markdown("""
    ## 🤖 AI-Powered Financial Planning
    
    This application uses machine learning to provide personalized financial planning recommendations 
    based on your income, expenses, savings, and financial goals.
    
    ### Features
    
    - **Personalized Recommendations**: Get AI-powered investment suggestions tailored to your financial situation
    - **Financial Health Analysis**: Understand your current financial health with detailed metrics
    - **Risk Assessment**: Recommendations based on your risk tolerance and investment experience
    - **Goal Planning**: Timeline analysis for achieving your financial goals
    - **Data Visualization**: Interactive charts and graphs to understand your financial patterns
    
    ### How It Works
    
    1. **Data Input**: Enter your financial information including income, expenses, savings, and goals
    2. **AI Analysis**: Our machine learning model analyzes your data using advanced algorithms
    3. **Personalized Plan**: Receive customized investment recommendations and financial insights
    4. **Actionable Insights**: Get specific recommendations to improve your financial health
    
    ### Technology Stack
    
    - **Machine Learning**: scikit-learn, pandas, numpy
    - **Web Interface**: Streamlit, Flask
    - **Visualization**: Plotly, matplotlib, seaborn
    - **Backend**: Python 3.x
    
    ### Disclaimer
    
    This application is for educational and informational purposes only. It should not be considered 
    as professional financial advice. Always consult with a qualified financial advisor before 
    making investment decisions.
    """)
    
    st.markdown("---")
    st.markdown("**Developed with ❤️ using Python and Machine Learning**")

def calculate_financial_health_score(savings_rate, expense_rate, debt_ratio):
    """Calculate a simple financial health score."""
    score = 0
    
    # Savings rate (40% of score)
    if savings_rate >= 20:
        score += 40
    elif savings_rate >= 10:
        score += 30
    elif savings_rate >= 5:
        score += 20
    else:
        score += 10
    
    # Expense control (30% of score)
    if expense_rate <= 50:
        score += 30
    elif expense_rate <= 70:
        score += 25
    elif expense_rate <= 80:
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

def generate_insights(user_data, prediction):
    """Generate financial insights."""
    income = user_data['income']
    expenses = user_data['expenses']
    savings = user_data['savings']
    goals = user_data['goals']
    recommended_plan = prediction['recommended_plan']
    
    savings_rate = (savings / income) * 100 if income > 0 else 0
    expense_rate = (expenses / income) * 100 if income > 0 else 0
    goal_achievement_time = goals / recommended_plan if recommended_plan > 0 else float('inf')
    
    recommendations = []
    
    if savings_rate < 10:
        recommendations.append("Consider increasing your savings rate. Aim for at least 20% of your income.")
    elif savings_rate > 30:
        recommendations.append("Great job! You have a healthy savings rate. Consider investing more aggressively.")
    
    if expense_rate > 80:
        recommendations.append("Your expenses are quite high relative to income. Look for ways to reduce costs.")
    
    if goal_achievement_time > 20:
        recommendations.append("Your goals may take a long time to achieve. Consider increasing your investment amount or adjusting your goals.")
    
    if not recommendations:
        recommendations.append("Your financial situation looks balanced. Keep up the good work!")
    
    return {
        'savings_rate': savings_rate,
        'expense_rate': expense_rate,
        'goal_achievement_time_years': goal_achievement_time,
        'recommendations': recommendations
    }

def get_investment_strategy(risk_tolerance):
    """Get investment strategy based on risk tolerance."""
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

if __name__ == "__main__":
    main()

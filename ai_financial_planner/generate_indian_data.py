#!/usr/bin/env python3
"""
Generate Indian Financial Dataset
This script creates a realistic financial dataset with Indian salary ranges and financial patterns.
"""

import pandas as pd
import numpy as np
import random

def generate_indian_financial_data():
    """Generate realistic financial data for Indian market."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate realistic financial data for Indian market
    n_samples = 1000
    
    # Income data (annual, in Lakhs of INR)
    # Indian salary ranges: 3L-50L INR annually
    incomes = np.random.normal(8, 3, n_samples)  # 8L average, 3L std
    incomes = np.clip(incomes, 3, 50)  # Cap between 3L-50L INR
    
    # Expenses (as percentage of income, 60-90%)
    expense_ratios = np.random.normal(0.75, 0.1, n_samples)
    expense_ratios = np.clip(expense_ratios, 0.6, 0.9)
    expenses = incomes * expense_ratios
    
    # Savings (remaining after expenses)
    savings = incomes - expenses
    savings = np.clip(savings, 0, incomes * 0.4)  # Max 40% savings
    
    # Financial goals (1.5x to 5x annual income) - typical Indian goals
    goals = incomes * np.random.uniform(1.5, 5, n_samples)
    
    # Calculate recommended plan (investment amount)
    # Based on income, expenses, savings, and goals
    recommended_plans = []
    for i in range(n_samples):
        # Base recommendation: 20-30% of savings for investment
        base_investment = savings[i] * np.random.uniform(0.2, 0.3)
        
        # Adjust based on goal size relative to income
        goal_ratio = goals[i] / incomes[i]
        if goal_ratio > 3:
            # High goals need more aggressive investment
            multiplier = np.random.uniform(1.2, 1.5)
        elif goal_ratio < 2:
            # Low goals, conservative approach
            multiplier = np.random.uniform(0.8, 1.0)
        else:
            # Moderate goals
            multiplier = np.random.uniform(1.0, 1.2)
        
        recommended_plan = base_investment * multiplier
        recommended_plans.append(max(0, recommended_plan))
    
    # Create DataFrame
    data = pd.DataFrame({
        'income': incomes,
        'expenses': expenses,
        'savings': savings,
        'goals': goals,
        'recommended_plan': recommended_plans
    })
    
    # Add some additional features for better modeling
    data['debt_ratio'] = np.random.uniform(0, 0.3, n_samples)  # Debt as % of income
    data['age'] = np.random.randint(25, 65, n_samples)
    data['risk_tolerance'] = np.random.choice(['low', 'medium', 'high'], n_samples)
    data['investment_experience'] = np.random.choice(['beginner', 'intermediate', 'advanced'], n_samples)
    
    # Save to CSV
    data.to_csv('data/user_data.csv', index=False)
    print('Indian Financial Dataset created successfully!')
    print(f'Shape: {data.shape}')
    print('\nSample data:')
    print(data.head())
    
    print('\nFinancial Statistics (in Lakhs INR):')
    print(f'Average Income: ₹{data["income"].mean():.2f}L')
    print(f'Average Expenses: ₹{data["expenses"].mean():.2f}L')
    print(f'Average Savings: ₹{data["savings"].mean():.2f}L')
    print(f'Average Goals: ₹{data["goals"].mean():.2f}L')
    print(f'Average Recommended Investment: ₹{data["recommended_plan"].mean():.2f}L')
    
    return data

if __name__ == "__main__":
    generate_indian_financial_data()

"""
AI Financial Chatbot
This module provides an intelligent chatbot that can answer financial planning questions
and clarify doubts about investment recommendations.
"""

import re
import random
from datetime import datetime

class FinancialChatbot:
    """
    An AI-powered chatbot that provides financial advice and clarifications
    based on user questions and their financial data.
    """
    
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! I'm your AI Financial Assistant. How can I help you with your financial planning today?",
                "Hi there! I'm here to help clarify any doubts about your financial planning. What would you like to know?",
                "Welcome! I can help answer your questions about investments, savings, and financial goals. What's on your mind?"
            ],
            'investment_questions': {
                'what_is_investment': "Investment is putting your money into assets like stocks, bonds, mutual funds, or real estate with the expectation of earning returns over time. It's different from savings because it involves some risk but potentially higher returns.",
                'how_much_invest': "The amount you should invest depends on your income, expenses, and financial goals. Generally, aim to invest 20-30% of your savings, but start with what you can afford and gradually increase.",
                'where_invest': "For beginners, consider: 1) Mutual Funds (diversified and managed), 2) PPF (tax-free, safe), 3) ELSS (tax-saving mutual funds). For experienced investors: Direct stocks, ETFs, or real estate.",
                'risk_tolerance': "Risk tolerance depends on your age, income stability, and comfort level. Young people can take more risk, while those near retirement should be more conservative. Your risk tolerance affects your investment choices.",
                'asset_allocation': "Asset allocation is key: 100-age = equity percentage, rest in debt and other assets. For example, at 30 years, consider 70% equity, 20% debt, 10% gold/real estate.",
                'portfolio_diversification': "Diversify across: 1) Asset classes (equity, debt, gold, real estate), 2) Market caps (large, mid, small-cap), 3) Sectors, 4) Geographies (consider international funds for 10-15% of portfolio).",
                'sip_vs_lumpsum': "SIP (Systematic Investment Plan) is better for most investors as it provides rupee cost averaging, reduces timing risk, and enforces discipline. Lump sum is better when markets are low.",
                'rebalancing': "Rebalance your portfolio annually or when allocation drifts by 5%. This maintains your target risk-return profile and helps lock in gains."
            },
            'savings_questions': {
                'emergency_fund': "An emergency fund should cover 3-6 months of your expenses. Keep it in a high-yield savings account or liquid mutual funds for easy access during emergencies.",
                'savings_rate': "A good savings rate is 20-30% of your income. If you're saving less, try the 50-30-20 rule: 50% needs, 30% wants, 20% savings and investments.",
                'increase_savings': "To increase savings: 1) Track your expenses, 2) Create a budget, 3) Automate savings, 4) Reduce unnecessary expenses, 5) Increase income through side hustles."
            },
            'goal_planning': {
                'short_term_goals': "Short-term goals (1-3 years) like vacation or emergency fund should be in safe investments like FDs, liquid funds, or high-yield savings accounts.",
                'long_term_goals': "Long-term goals (5+ years) like retirement or buying a house can include equity mutual funds, PPF, or real estate for better returns.",
                'goal_prioritization': "Prioritize goals by: 1) Emergency fund first, 2) High-interest debt repayment, 3) Retirement planning, 4) Other goals based on timeline and importance.",
                'goal_based_investing': "Match investment horizon with goal timeline: 1-3 years (debt funds), 3-7 years (balanced funds), 7+ years (equity funds). Consider inflation (6-7% annually) for long-term goals.",
                'retirement_planning': "For retirement: 1) Start early with SIPs, 2) Use EPF/PPF for tax benefits, 3) Consider NPS for additional corpus, 4) Plan for healthcare costs, 5) Aim for 25-30x annual expenses as retirement corpus.",
                'education_planning': "For child's education: 1) Start SIP in equity funds when child is young, 2) Switch to debt funds 2-3 years before need, 3) Consider education loans for flexibility, 4) Use Sukanya Samriddhi for daughters."
            },
            'portfolio_management': {
                'tracking_performance': "Track portfolio using XIRR for accurate returns. Review quarterly, rebalance annually or when allocation drifts by 5%. Use portfolio tracking tools for monitoring.",
                'diversification_strategy': "Diversify across: 1) Asset classes (equity 60-80%, debt 20-30%, gold 5-10%), 2) Market caps (large 40%, mid 30%, small 30%), 3) Sectors, 4) Geographies (10-15% international).",
                'risk_management': "Manage risk through: 1) Diversification, 2) Asset allocation, 3) Regular rebalancing, 4) Stop-loss for direct equity, 5) Emergency fund, 6) Insurance coverage.",
                'tax_optimization': "Optimize taxes through: 1) ELSS for 80C benefits, 2) Long-term capital gains (1 year+ for equity), 3) Tax-loss harvesting, 4) PPF/NPS for additional tax benefits."
            },
            'indian_specific': {
                'tax_saving': "In India, you can save tax through: 1) ELSS mutual funds (₹1.5L under 80C), 2) PPF (₹1.5L), 3) NPS (₹50K under 80CCD), 4) Health insurance (₹25K under 80D).",
                'inflation': "In India, inflation is around 6-7% annually. Your investments should beat inflation to maintain purchasing power. Equity investments historically outperform inflation.",
                'retirement_planning': "For retirement in India: 1) Start early with SIPs, 2) Use EPF/PPF for tax benefits, 3) Consider NPS for additional retirement corpus, 4) Plan for healthcare costs."
            },
            'clarification': [
                "Let me clarify that for you. Based on your financial situation, here's what I recommend...",
                "That's a great question! Let me break it down for you...",
                "I understand your concern. Here's a detailed explanation...",
                "Let me provide more context to help you understand better..."
            ],
            'fallback': [
                "I understand you have a question about that. Could you be more specific so I can help you better?",
                "That's an interesting point. Let me know more details about your specific situation.",
                "I'd be happy to help! Could you rephrase your question or provide more context?",
                "I want to make sure I give you the best advice. Can you tell me more about your specific concern?"
            ]
        }
    
    def analyze_question(self, question, user_data=None):
        """
        Analyze the user's question and provide appropriate response.
        
        Args:
            question (str): User's question
            user_data (dict): User's financial data for personalized advice
        
        Returns:
            dict: Response with answer and suggestions
        """
        question_lower = question.lower()
        
        # Detect question type
        if any(word in question_lower for word in ['hello', 'hi', 'hey', 'start']):
            return self._handle_greeting()
        
        elif any(word in question_lower for word in ['invest', 'investment', 'investing']):
            return self._handle_investment_question(question_lower, user_data)
        
        elif any(word in question_lower for word in ['save', 'saving', 'savings']):
            return self._handle_savings_question(question_lower, user_data)
        
        elif any(word in question_lower for word in ['goal', 'goals', 'planning']):
            return self._handle_goal_question(question_lower, user_data)
        
        elif any(word in question_lower for word in ['tax', 'ppf', 'elss', 'nps', 'inflation']):
            return self._handle_indian_specific_question(question_lower, user_data)
        
        elif any(word in question_lower for word in ['explain', 'clarify', 'doubt', 'confused', 'understand']):
            return self._handle_clarification_request(question_lower, user_data)
        
        else:
            return self._handle_general_question(question, user_data)
    
    def _handle_greeting(self):
        """Handle greeting messages."""
        return {
            'type': 'greeting',
            'message': random.choice(self.responses['greeting']),
            'suggestions': [
                "What should I invest in?",
                "How much should I save?",
                "Explain my financial plan",
                "What are good investment options in India?"
            ]
        }
    
    def _handle_investment_question(self, question, user_data):
        """Handle investment-related questions."""
        if 'what is' in question or 'what are' in question:
            key = 'what_is_investment'
        elif 'how much' in question:
            key = 'how_much_invest'
        elif 'where' in question or 'which' in question:
            key = 'where_invest'
        elif 'risk' in question:
            key = 'risk_tolerance'
        else:
            key = 'what_is_investment'
        
        response = self.responses['investment_questions'][key]
        
        # Add personalized advice if user data is available
        if user_data:
            income = user_data.get('income', 0)
            savings = user_data.get('savings', 0)
            risk_tolerance = user_data.get('risk_tolerance', 'medium')
            
            if key == 'how_much_invest':
                recommended_amount = savings * 0.25  # 25% of savings
                response += f"\n\nBased on your current savings of ₹{savings:.1f}L, I recommend investing around ₹{recommended_amount:.1f}L annually."
            
            if key == 'where_invest':
                if risk_tolerance == 'low':
                    response += "\n\nGiven your low risk tolerance, focus on: PPF, FDs, and debt mutual funds."
                elif risk_tolerance == 'high':
                    response += "\n\nWith your high risk tolerance, you can consider: Direct stocks, equity mutual funds, and real estate."
        
        return {
            'type': 'investment',
            'message': response,
            'suggestions': [
                "What's my risk tolerance?",
                "How to start investing?",
                "Best mutual funds for beginners",
                "Explain SIP vs lump sum"
            ]
        }
    
    def _handle_savings_question(self, question, user_data):
        """Handle savings-related questions."""
        if 'emergency' in question:
            key = 'emergency_fund'
        elif 'rate' in question or 'percentage' in question:
            key = 'savings_rate'
        elif 'increase' in question or 'more' in question:
            key = 'increase_savings'
        else:
            key = 'savings_rate'
        
        response = self.responses['savings_questions'][key]
        
        # Add personalized advice
        if user_data:
            income = user_data.get('income', 0)
            expenses = user_data.get('expenses', 0)
            savings = user_data.get('savings', 0)
            
            if income > 0:
                current_savings_rate = (savings / income) * 100
                response += f"\n\nYour current savings rate is {current_savings_rate:.1f}%. "
                
                if current_savings_rate < 20:
                    response += "Consider increasing it to at least 20% for better financial health."
                elif current_savings_rate > 30:
                    response += "Great job! You have an excellent savings rate."
        
        return {
            'type': 'savings',
            'message': response,
            'suggestions': [
                "How to build emergency fund?",
                "50-30-20 rule explained",
                "Best savings accounts in India",
                "How to increase my savings?"
            ]
        }
    
    def _handle_goal_question(self, question, user_data):
        """Handle goal planning questions."""
        if 'short' in question or 'immediate' in question:
            key = 'short_term_goals'
        elif 'long' in question or 'retirement' in question:
            key = 'long_term_goals'
        elif 'priorit' in question:
            key = 'goal_prioritization'
        else:
            key = 'goal_prioritization'
        
        response = self.responses['goal_planning'][key]
        
        # Add personalized advice
        if user_data:
            goals = user_data.get('goals', 0)
            income = user_data.get('income', 0)
            
            if goals > 0 and income > 0:
                goal_ratio = goals / income
                response += f"\n\nYour goal of ₹{goals:.1f}L is {goal_ratio:.1f}x your annual income. "
                
                if goal_ratio > 5:
                    response += "This is quite ambitious - consider breaking it into smaller, achievable milestones."
                elif goal_ratio < 2:
                    response += "This is a reasonable goal that you can achieve with consistent saving and investing."
        
        return {
            'type': 'goals',
            'message': response,
            'suggestions': [
                "How to prioritize financial goals?",
                "Short vs long term investment strategies",
                "Retirement planning in India",
                "Goal-based investing explained"
            ]
        }
    
    def _handle_indian_specific_question(self, question, user_data):
        """Handle India-specific financial questions."""
        if 'tax' in question:
            key = 'tax_saving'
        elif 'inflation' in question:
            key = 'inflation'
        elif 'retirement' in question:
            key = 'retirement_planning'
        else:
            key = 'tax_saving'
        
        response = self.responses['indian_specific'][key]
        
        return {
            'type': 'indian_specific',
            'message': response,
            'suggestions': [
                "Best tax saving investments",
                "PPF vs ELSS comparison",
                "NPS benefits explained",
                "Inflation impact on investments"
            ]
        }
    
    def _handle_clarification_request(self, question, user_data):
        """Handle requests for clarification."""
        response = random.choice(self.responses['clarification'])
        
        # Try to identify what needs clarification
        if 'investment' in question:
            response += " Regarding investments, the key is to start early, diversify your portfolio, and stay consistent with your SIPs."
        elif 'savings' in question:
            response += " For savings, focus on building an emergency fund first, then work on increasing your savings rate gradually."
        elif 'goal' in question:
            response += " For goal planning, prioritize based on timeline and importance, and choose appropriate investment vehicles."
        else:
            response += " Please let me know which specific aspect you'd like me to explain in more detail."
        
        return {
            'type': 'clarification',
            'message': response,
            'suggestions': [
                "Explain step by step",
                "Give me an example",
                "What are the risks?",
                "How to get started?"
            ]
        }
    
    def _handle_general_question(self, question, user_data):
        """Handle general questions."""
        response = random.choice(self.responses['fallback'])
        
        return {
            'type': 'general',
            'message': response,
            'suggestions': [
                "Investment basics",
                "Savings strategies",
                "Goal planning",
                "Tax saving options"
            ]
        }
    
    def get_personalized_advice(self, user_data, question_type):
        """Get personalized advice based on user's financial data."""
        income = user_data.get('income', 0)
        expenses = user_data.get('expenses', 0)
        savings = user_data.get('savings', 0)
        goals = user_data.get('goals', 0)
        risk_tolerance = user_data.get('risk_tolerance', 'medium')
        
        advice = []
        
        # Analyze financial health
        if income > 0:
            savings_rate = (savings / income) * 100
            expense_rate = (expenses / income) * 100
            
            if savings_rate < 10:
                advice.append("⚠️ Your savings rate is quite low. Try to increase it to at least 20% of your income.")
            elif savings_rate > 30:
                advice.append("✅ Excellent savings rate! You're on track for a secure financial future.")
            
            if expense_rate > 80:
                advice.append("⚠️ Your expenses are very high relative to income. Look for ways to reduce costs.")
        
        # Risk tolerance advice
        if risk_tolerance == 'low':
            advice.append("💡 With low risk tolerance, focus on safe investments like PPF, FDs, and debt mutual funds.")
        elif risk_tolerance == 'high':
            advice.append("💡 With high risk tolerance, you can consider equity investments for better long-term returns.")
        
        # Goal analysis
        if goals > 0 and income > 0:
            goal_ratio = goals / income
            if goal_ratio > 5:
                advice.append("🎯 Your financial goal is quite ambitious. Consider breaking it into smaller milestones.")
            elif goal_ratio < 2:
                advice.append("🎯 Your goal is achievable with consistent saving and investing.")
        
        return advice

# Initialize the chatbot
chatbot = FinancialChatbot()

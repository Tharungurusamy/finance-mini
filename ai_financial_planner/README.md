# 🤖 AI Financial Planning Assistant

An intelligent financial planning application that uses machine learning to provide personalized investment recommendations based on user financial data.

## 🌟 Features

- **AI-Powered Recommendations**: Get personalized investment suggestions using machine learning algorithms
- **Financial Health Analysis**: Comprehensive analysis of your financial situation with health scoring
- **Multiple Interfaces**: Both Flask web API and Streamlit dashboard for different use cases
- **Interactive Visualizations**: Beautiful charts and graphs to understand your financial patterns
- **Risk Assessment**: Recommendations tailored to your risk tolerance and investment experience
- **Goal Planning**: Timeline analysis for achieving your financial objectives
- **Real-time Predictions**: Instant recommendations based on your input data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd ai_financial_planner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time only)
   ```bash
   python model/financial_model.py
   ```

### Running the Application

#### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

#### Option 2: Flask Web API
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

## 📁 Project Structure

```
ai_financial_planner/
│
├── app.py                  # Flask web API
├── streamlit_app.py       # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── model/
│   └── financial_model.py # ML model training and prediction
│
├── data/
│   └── user_data.csv     # Sample financial dataset
│
├── templates/
│   └── index.html        # Flask web interface
│
└── static/               # Static assets (generated)
```

## 🔧 Usage

### Input Data Format

The application expects the following financial information:

- **income**: Annual income (in dollars)
- **expenses**: Annual expenses (in dollars)
- **savings**: Annual savings (in dollars)
- **goals**: Financial goal amount (in dollars)
- **debt_ratio**: Debt as percentage of income (0-1)
- **risk_tolerance**: "low", "medium", or "high"
- **investment_experience**: "beginner", "intermediate", or "advanced"

### Example API Usage

```python
import requests

# Example API call
data = {
    "income": 75000,
    "expenses": 50000,
    "savings": 15000,
    "goals": 200000,
    "debt_ratio": 0.1,
    "risk_tolerance": "medium",
    "investment_experience": "intermediate"
}

response = requests.post('http://localhost:5000/api/predict', json=data)
result = response.json()

print(f"Recommended Investment: ${result['prediction']['recommended_plan']:,.2f}")
print(f"Confidence: {result['prediction']['confidence']:.1%}")
```

### Example JSON Response

```json
{
  "success": true,
  "prediction": {
    "recommended_plan": 12500.0,
    "confidence": 0.85,
    "model_type": "RandomForest"
  },
  "insights": {
    "savings_rate": 20.0,
    "expense_ratio": 66.7,
    "goal_achievement_time_years": 16.0,
    "recommendations": [
      "Your financial situation looks balanced. Keep up the good work!"
    ]
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🧠 Machine Learning Model

### Model Architecture

The application uses a **Random Forest Regressor** as the primary model, with support for:
- **Gradient Boosting Regressor**
- **Linear Regression**

### Features Used

- Income, expenses, savings, and goals
- Debt ratio
- Risk tolerance (encoded)
- Investment experience (encoded)

### Model Training

The model is trained on a synthetic dataset of 1,000 financial profiles with realistic patterns:

```python
# Train the model
from model.financial_model import train_and_save_model
model = train_and_save_model()
```

### Model Performance

The model typically achieves:
- **R² Score**: 0.85+ (explains 85% of variance)
- **RMSE**: Low error rate for investment recommendations
- **Cross-validation**: Robust performance across different data splits

## 📊 API Endpoints

### Flask API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard page |
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Get financial recommendations |
| `/api/insights` | POST | Get detailed financial insights |
| `/api/model/info` | GET | Model information |

### Example API Calls

```bash
# Health check
curl http://localhost:5000/api/health

# Get prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 75000, "expenses": 50000, "savings": 15000, "goals": 200000}'
```

## 🎯 Financial Insights

The application provides several types of insights:

### 1. Financial Health Score
- Calculated based on savings rate, expense control, and debt management
- Scale: 0-100 (higher is better)

### 2. Investment Strategy
- **Conservative (Low Risk)**: 70% Bonds, 20% Stocks, 10% Cash
- **Balanced (Medium Risk)**: 40% Bonds, 50% Stocks, 10% Cash
- **Aggressive (High Risk)**: 20% Bonds, 70% Stocks, 10% Cash

### 3. Goal Timeline Analysis
- Estimates time to achieve financial goals
- Provides recommendations for goal adjustment

### 4. Personalized Recommendations
- Actionable advice based on financial situation
- Specific steps to improve financial health

## 🔍 Data Analysis Features

The Streamlit dashboard includes:

- **Dataset Overview**: Statistical summary of financial data
- **Income Distribution**: Histogram of income patterns
- **Savings Analysis**: Correlation between income and savings
- **Risk Tolerance Analysis**: Investment patterns by risk level
- **Feature Correlation**: Heatmap of financial feature relationships

## 🛠️ Customization

### Adding New Features

1. **Extend the dataset** in `data/user_data.csv`
2. **Update feature columns** in `FinancialPlannerModel.preprocess_data()`
3. **Retrain the model** using `python model/financial_model.py`

### Modifying the Model

```python
# Use different model types
model.train_model('GradientBoosting')  # or 'LinearRegression'

# Adjust model parameters
model = RandomForestRegressor(
    n_estimators=200,  # More trees
    max_depth=15,      # Deeper trees
    random_state=42
)
```

### Customizing Insights

Modify the insight generation functions in `app.py`:
- `generate_insights()`
- `calculate_financial_health_score()`
- `get_investment_strategy()`

## 🧪 Testing

### Test the Model

```python
# Test with sample data
python model/financial_model.py
```

### Test the API

```bash
# Start the Flask app
python app.py

# In another terminal, test the API
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 75000, "expenses": 50000, "savings": 15000, "goals": 200000}'
```

### Test the Streamlit App

```bash
streamlit run streamlit_app.py
```

## 📈 Performance Optimization

### Model Optimization

- **Feature Engineering**: Add more relevant financial features
- **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
- **Ensemble Methods**: Combine multiple models for better predictions

### Application Optimization

- **Caching**: Use `@st.cache_data` for expensive operations
- **Async Processing**: Implement async API endpoints for better performance
- **Database Integration**: Store user data and predictions persistently

## 🚀 Deployment

### Local Deployment

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python model/financial_model.py
   ```

3. **Run the application**
   ```bash
   # Streamlit
   streamlit run streamlit_app.py
   
   # Flask
   python app.py
   ```

### Cloud Deployment

#### Heroku
1. Create `Procfile`:
   ```
   web: python app.py
   ```

2. Deploy:
   ```bash
   git add .
   git commit -m "Deploy AI Financial Planner"
   git push heroku main
   ```

#### Docker
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

2. Build and run:
   ```bash
   docker build -t ai-financial-planner .
   docker run -p 5000:5000 ai-financial-planner
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It should not be considered as professional financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- **scikit-learn** for machine learning algorithms
- **Streamlit** for the interactive dashboard
- **Flask** for the web API
- **Plotly** for beautiful visualizations
- **Pandas** for data manipulation

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Contact the development team

---

**Built with ❤️ using Python and Machine Learning**

# Customer Churn Prediction

Machine learning project to predict customer churn for a telecommunications company using Random Forest classification and customer segmentation analysis.

## Project Overview
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 80%+
- **Dataset**: 7,000+ telecom customers
- **Tools**: Python, scikit-learn, pandas, Plotly, seaborn

## Features
- Customer segmentation using K-means clustering (4 segments)
- Interactive dashboards for risk visualization
- Automated recommendations for high-risk customers
- Model monitoring and drift detection system
- Feature importance analysis

## Key Findings
- Month-to-month contracts have highest churn risk
- Fiber optic customers without tech support are vulnerable
- Early tenure customers (0-12 months) need retention focus
- Contract type and tenure are the strongest churn predictors

## Dataset
The dataset is from [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and contains 7,043 customers with 21 features including:
- Demographics (gender, senior citizen status, partner, dependents)
- Services (phone, internet, security, backup, tech support, streaming)
- Account information (tenure, contract type, payment method, charges)

## Files
- `customer_churn_prediction.ipynb` - Main analysis notebook with all code and visualizations
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Source dataset
- `high_risk_customers.csv` - Identified at-risk customers for retention campaigns
- `churn_prediction_model.pkl` - Saved Random Forest model

## How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn plotly joblib

# Run the notebook
jupyter notebook customer_churn_prediction.ipynb

Results

Accuracy: 80%+
AUC-ROC: 0.84+
F1-Score: Balanced precision and recall for churn prediction
Successfully identified 4 distinct customer segments with varying churn rates
Generated actionable retention strategies for 500+ high-risk customers

Model Performance Metrics

Confusion Matrix analysis showing true/false positives and negatives
ROC Curve demonstrating strong discrimination between churners and non-churners
Feature importance rankings identifying key churn drivers

Project Structure

customer-churn-prediction/
├── customer_churn_prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── high_risk_customers.csv
├── churn_prediction_model.pkl
├── model_logs/
│   ├── performance_log.json
│   └── training_stats.json
└── README.md

Technologies Used

Python 3.9+
scikit-learn: Machine learning algorithms and preprocessing
pandas & NumPy: Data manipulation and analysis
Matplotlib & Seaborn: Statistical visualizations
Plotly: Interactive dashboards
joblib: Model serialization

Methodology

Data Exploration: Analyzed customer demographics, services, and churn patterns
Feature Engineering: Created tenure groups, price sensitivity, and service count features
Model Training: Implemented Random Forest with hyperparameter tuning
Evaluation: Assessed performance using multiple metrics (accuracy, precision, recall, AUC)
Segmentation: Applied K-means clustering to identify customer segments
Deployment: Built prediction pipeline with monitoring capabilities

Future Improvements

Implement real-time prediction API using Flask or FastAPI
Add more sophisticated feature engineering (customer lifetime value, usage trends)
Test additional algorithms (XGBoost, LightGBM, Neural Networks)
Integrate with CRM systems for automated retention alerts
Deploy model monitoring dashboard for production use

License
This project is open source and available for educational purposes.

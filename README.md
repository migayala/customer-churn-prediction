# Customer Churn Prediction

Machine learning project to predict customer churn for a telecommunications company.

## Project Overview
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 80%+
- **Dataset**: 7,000+ telecom customers
- **Tools**: Python, scikit-learn, pandas, Plotly

## Features
- Customer segmentation using K-means clustering
- Interactive dashboards for risk visualization
- Automated recommendations for high-risk customers
- Model monitoring and drift detection

## Key Findings
- Month-to-month contracts have highest churn risk
- Fiber optic customers without tech support are vulnerable
- Early tenure customers (0-12 months) need retention focus

## Files
- `customer_churn_prediction.ipynb` - Main analysis notebook
- `high_risk_customers.csv` - Identified at-risk customers

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
jupyter notebook customer_churn_prediction.ipynb

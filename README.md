# Vehicle Insurance Claim Fraud Detection

## Overview
Built a supervised machine learning pipeline to detect fraudulent vehicle 
insurance claims, using the real Oracle fraud detection dataset 
(Kaggle: shivamb/vehicle-claim-fraud-detection). The goal is to help an 
insurance company prioritise which claims to investigate - catching the 
most fraud while managing investigation costs.

---

## Business Problem
Insurance fraud costs the UK industry an estimated £1.1 billion every year. 
Most claims are genuine, fraud is rare (~6%), and missing a fraud is far more 
expensive than investigating a false alarm. This project builds a model that 
scores every claim by fraud risk so investigators can focus their time where 
it matters most.

---

## Dataset
- Source   : Kaggle - shivamb/vehicle-claim-fraud-detection
- Rows     : 15,420 claims
- Columns  : 33 features (demographics, vehicle, policy, claim details)
- Target   : FraudFound_P (0 = genuine, 1 = fraud)
- Fraud rate : ~6% - a realistic, heavily imbalanced classification problem

---

## What I Built
- Cleaned real-world data: removed ID columns, handled Age coded as 0 
  (missing), auto-detected numeric vs categorical features
- Compared a Logistic Regression baseline against a Random Forest using 
  stratified 5-fold cross-validation
- Evaluated using ROC-AUC and PR-AUC - not accuracy, which is meaningless 
  at 6% fraud prevalence
- Selected a decision threshold based on business cost (missed fraud vs 
  investigation cost) rather than the default 0.5
- Visualised feature importance to explain which signals drive fraud risk

---

## Results

| Metric              | Score  | Notes                          |
|---------------------|--------|--------------------------------|
| ROC-AUC             | 0.826  | Strong ranking on a hard dataset|
| PR-AUC              | 0.221  | vs 0.060 no-skill baseline     |
| Improvement         | 3.7x   | Better than random at fraud    |
| Fraud recall        | 72.3%  | At cost-optimal threshold 0.33 |

---

## Key Findings
- Fault (who caused the accident) is the single strongest fraud signal
- Liability policy types carry significantly higher fraud risk
- Urban accident areas show higher fraud rates than rural
- At a cost ratio of K=10 (missing fraud = 10x an investigation), the model recommends flagging aggressively to maximise recall

---

## Tech Stack
- Python 3
- pandas - data loading and cleaning
- scikit-learn - preprocessing, modelling, evaluation
- matplotlib / seaborn - visualisation
- Google Colab - development environment

---
## Project Structure

```
fraud-detection/
├── vehicle_fraud_detection.ipynb   # full analysis with outputs
├── fraud_oracle.csv                # dataset (source: Kaggle)
└── README.md                       # this file
```

---

## How to Run
1. Clone this repository
2. Open vehicle_fraud_detection.ipynb in Google Colab or Jupyter
3. Run all cells from top to bottom
4. All outputs and charts generate automatically

---

## Skills Demonstrated
- Real-world data cleaning on a messy dataset
- Handling severe class imbalance (stratified splits, class_weight)
- Model comparison with cross-validation
- Choosing evaluation metrics appropriate to the problem (PR-AUC over accuracy)
- Translating model output into a business decision via cost-based 
  threshold selection
- Feature importance interpretation tied to domain knowledge

---

Author: Shreya Johar
MSc Business Analytics - Warwick Business School
GitHub: github.com/Shreyajohar2004

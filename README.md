# 🔍 Payment Fraud Detection System
A machine learning system that detects fraudulent credit card transactions in real time using anomaly detection and supervised learning.

**Live Demo →** (https://upi-fraud-detection-n7dxwhzuyyh893yj9phm87.streamlit.app/)

---

## 🎯 Problem Statement

Credit card fraud is a core problem for every financial institution and payment processor.
Every day, banks and fintech companies process millions of transactions — and fraudsters exploit speed, anonymity, and behavioral patterns to steal money.

This project builds a fraud scoring system that flags suspicious transactions before they cause damage.

---

## 🚨 Fraud Patterns Detected

| Pattern | Description |
|---|---|
| Spike anomaly | Amount unusually high for that hour |
| Velocity fraud | Too many transactions in short time |
| Night transactions | Payments between 11pm and 5am |
| Salami slicing | Suspiciously small repeated amounts |

---

## 🛠️ Tech Stack

- **Language** — Python
- **ML Models** — Random Forest, Logistic Regression, Isolation Forest
- **Imbalance Handling** — SMOTE (Synthetic Minority Oversampling)
- **Dashboard** — Streamlit + Plotly
- **Dataset** — Credit Card Fraud Detection (Kaggle, 284,807 transactions)

---

## 📊 Model Performance

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Random Forest | 0.80 | 0.88 | 0.83 |
| Logistic Regression | 0.48 | 0.91 | 0.14 |
| Isolation Forest | 0.21 | 0.24 | 0.23 |

**Chosen model — Random Forest** because it achieves the best balance between catching fraud (recall 88%) and avoiding false alarms (precision 80%).

---

## ⚙️ Feature Engineering

Features engineered from raw transaction data:

- `amount_zscore` — how unusual is this amount for this hour
- `amount_log` — log-normalized amount to reduce skew
- `is_night` — binary flag for 11pm to 5am transactions
- `velocity_10` — transaction count in current time window

---

## ⚠️ Known Limitations

The dataset uses PCA-anonymized features (V1–V28) which cannot be reproduced for new manual inputs. In production, raw features like merchant ID, device fingerprint, and GPS location would replace these. The live transaction checker uses engineered features only and is intended as a demonstration.

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run dashboard
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                                       # Streamlit dashboard
├── fraud_results.csv                            # Pre-scored transaction results
├── fraud_model.pkl                              # Trained Random Forest model
├── scaler.pkl                                   # Fitted StandardScaler
├── Credit Card Fraud Detection.ipynb            # Jupyter notebook
├── requirements.txt                             # Dependencies
└── README.md                                    # You are here
```

---

## 💡 Key Learnings

- Class imbalance (0.17% fraud) is the #1 challenge in fraud detection
- SMOTE must be applied after train/test split to avoid data leakage
- Recall matters more than precision in fraud detection — missing fraud is costlier than a false alarm
- Threshold tuning (0.3 vs 0.5) is a business decision, not a technical one

---

## 🔮 Future Improvements

- [ ] Add real-world transaction features (merchant category, device, location)
- [ ] Implement XGBoost for better performance
- [ ] Add time-series analysis of spending patterns
- [ ] Build user-level behavioral baseline model
- [ ] Add explainability using SHAP values

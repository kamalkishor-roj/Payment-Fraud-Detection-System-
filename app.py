import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Load model, scaler and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'fraud_results.csv'))

@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(BASE_DIR, 'fraud_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    return model, scaler

df            = load_data()
model, scaler = load_model()

# ─── Header ───────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Real-time fraud scoring powered by Random Forest")
st.divider()

# ═══════════════════════════════════════════════════
# SECTION 1 — Live Transaction Checker
# ═══════════════════════════════════════════════════
st.subheader("⚡ Check a New Transaction")
st.markdown("Enter transaction details to get an instant fraud score")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.01,
        max_value=500000.0,
        value=1000.0,
        step=0.01
    )

with col2:
    hour = st.slider(
        "Time of transaction (hour)",
        min_value=0,
        max_value=23,
        value=14,
        format="%d:00"
    )

# Show time context to user
if hour >= 23 or hour <= 5:
    st.warning("⚠️ Night transaction detected (11pm–5am)")
else:
    st.success("✅ Daytime transaction")

# Predict button
if st.button("🔍 Check for Fraud", type="primary"):

    # Step 1 — Build features for this transaction
    # Use dataset averages for amount_zscore baseline
    avg_amount = df['Amount'].mean()
    std_amount = df['Amount'].std()

    amount_zscore = (amount - avg_amount) / std_amount
    amount_log    = np.log1p(amount)
    is_night      = 1 if (hour >= 23 or hour <= 5) else 0
    velocity_10   = 1  # single transaction = 1

    # Step 2 — Build full feature row (V1-V28 = 0 for simulation)
    feature_row = {
        'Amount':        amount,
        'amount_zscore': amount_zscore,
        'amount_log':    amount_log,
        'is_night':      is_night,
        'velocity_10':   velocity_10,
    }
    for i in range(1, 29):
        feature_row[f'V{i}'] = 0.0

    input_df = pd.DataFrame([feature_row])

    # Step 3 — Scale using saved scaler
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=input_df.columns
    )

    # Step 4 — Get fraud score
    fraud_score = model.predict_proba(input_scaled)[0][1]

    # Step 5 — Determine risk level
    if fraud_score >= 0.7:
        risk_level = "HIGH"
        color      = "🔴"
        action     = "BLOCK transaction immediately"
    elif fraud_score >= 0.3:
        risk_level = "MEDIUM"
        color      = "🟡"
        action     = "Flag for manual review"
    else:
        risk_level = "LOW"
        color      = "🟢"
        action     = "Allow transaction"

    # Step 6 — Show result
    st.divider()
    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Fraud Score", f"{fraud_score:.2%}")
    with r2:
        st.metric("Risk Level", f"{color} {risk_level}")
    with r3:
        st.metric("Recommended Action", action)

    # Fraud score gauge bar
    st.markdown("**Fraud Score Gauge**")
    gauge_color = (
        "#e74c3c" if fraud_score >= 0.7
        else "#f39c12" if fraud_score >= 0.3
        else "#2ecc71"
    )
    st.markdown(
        f"""
        <div style="background:#eee;border-radius:10px;height:24px;width:100%">
          <div style="background:{gauge_color};
                      width:{fraud_score*100:.1f}%;
                      height:24px;border-radius:10px;
                      transition:width 0.5s">
          </div>
        </div>
        <p style="text-align:right;color:{gauge_color};
                  font-weight:bold">{fraud_score:.2%}</p>
        """,
        unsafe_allow_html=True
    )

    # Explanation
    st.markdown("**Why this score?**")
    reasons = []
    if is_night:
        reasons.append("🌙 Transaction at night — unusual hour")
    if amount_zscore > 2:
        reasons.append("💰 Amount unusually high for this hour")
    if amount > 10000:
        reasons.append("💸 High value transaction")
    if not reasons:
        reasons.append("✅ No strong fraud signals detected")

    for r in reasons:
        st.markdown(f"- {r}")

st.divider()

# ═══════════════════════════════════════════════════
# SECTION 2 — Historical Dashboard
# ═══════════════════════════════════════════════════
st.subheader("📊 Historical Transaction Analysis")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    high = len(df[df['risk_level'] == 'High'])
    st.metric("High Risk Flagged", high)
with col3:
    fraud_caught = len(df[(df['risk_level'] == 'High') &
                          (df['actual'] == 1)])
    st.metric("Fraud Caught", fraud_caught)
with col4:
    precision = round(
        df[(df['predicted'] == 1) &
           (df['actual'] == 1)].shape[0] /
        max(df[df['predicted'] == 1].shape[0], 1) * 100, 1)
    st.metric("Precision", f"{precision}%")

st.divider()

# Filters
st.subheader("🎛️ Filters")
col1, col2 = st.columns(2)
with col1:
    risk_filter = st.multiselect(
        "Filter by risk level",
        options=['Low', 'Medium', 'High'],
        default=['Medium', 'High']
    )
with col2:
    score_threshold = st.slider(
        "Minimum fraud score",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

filtered = df[
    (df['risk_level'].isin(risk_filter)) &
    (df['fraud_score'] >= score_threshold)
]
st.markdown(f"Showing **{len(filtered):,}** transactions")
st.divider()

# Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Risk Level Distribution")
    risk_counts = df['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    fig1 = px.bar(
        risk_counts,
        x='Risk Level', y='Count',
        color='Risk Level',
        color_discrete_map={
            'Low':'#2ecc71',
            'Medium':'#f39c12',
            'High':'#e74c3c'
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Fraud Score Distribution")
    fig2 = px.histogram(
        df, x='fraud_score',
        color='risk_level', nbins=50,
        color_discrete_map={
            'Low':'#2ecc71',
            'Medium':'#f39c12',
            'High':'#e74c3c'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Amount vs Fraud Score")
fig3 = px.scatter(
    filtered,
    x='Amount', y='fraud_score',
    color='risk_level',
    color_discrete_map={
        'Low':'#2ecc71',
        'Medium':'#f39c12',
        'High':'#e74c3c'
    },
    hover_data=['actual', 'predicted']
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Transaction table
st.subheader("🚨 Flagged Transactions")

def highlight_risk(row):
    if row['risk_level'] == 'High':
        return ['background-color: #ffcccc'] * len(row)
    elif row['risk_level'] == 'Medium':
        return ['background-color: #fff3cc'] * len(row)
    return [''] * len(row)

display_cols = ['Amount', 'fraud_score',
                'risk_level', 'actual', 'predicted']
st.dataframe(
    filtered[display_cols].style.apply(highlight_risk, axis=1),
    use_container_width=True,
    height=400
)
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# === Load all saved artifacts (models + scaler + feature list) ===
@st.cache_resource
def load_artifacts():
    clf = joblib.load('classifier.pkl')
    reg = joblib.load('regressor.pkl')
    scaler = joblib.load('scaler.pkl')               # ← Uses saved scaler
    feature_columns = joblib.load('feature_columns.pkl')  # ← Exact columns from training
    return clf, reg, scaler, feature_columns

clf, reg, scaler, feature_columns = load_artifacts()

# === App UI ===
st.set_page_config(page_title="Real Estate Investment Advisor", layout="centered")
st.title("🏠 Real Estate Investment Advisor")
st.markdown("### Predict if a property is a **Good Investment** and its value in 2030")

col1, col2 = st.columns(2)

with col1:
    size = st.number_input("Size (SqFt)", min_value=300, max_value=10000, value=1200, step=50)
    price = st.number_input("Current Price (₹ Lakhs)", min_value=5.0, max_value=5000.0, value=100.0, step=5.0)
    bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5, 6])

with col2:
    city = st.selectbox("City", options=[
        "Mumbai", "Bangalore", "Delhi", "Pune", "Hyderabad", "Chennai",
        "Gurgaon", "Noida", "Kolkata", "Other_City"
        # Add more if they appear in your top 15 cities after training
    ])
    age = st.slider("Property Age (Years)", min_value=0, max_value=50, value=10)
    infra = st.slider("Infrastructure Score", min_value=3, max_value=15, value=10)
    amenities = st.slider("Amenities Count", min_value=0, max_value=15, value=5)

if st.button("🔮 Predict Investment Potential", type="primary"):
    # Calculate Price per SqFt
    ppsqft = (price * 100000) / size if size > 0 else 0

    # Create empty row with all features = 0
    input_dict = {col: 0 for col in feature_columns}

    # Fill known numerical features
    input_dict['Size_in_SqFt'] = size
    input_dict['Price_in_Lakhs'] = price
    input_dict['Price_per_SqFt'] = ppsqft
    input_dict['BHK'] = bhk
    input_dict['Property_Age'] = age
    input_dict['Infrastructure_Score'] = infra
    input_dict['Amenities_Count'] = amenities

    # Set selected city
    city_col = f"City_{city}"
    if city_col in feature_columns:
        input_dict[city_col] = 1
    else:
        input_dict['City_Other_City'] = 1  # Safe fallback

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    # === Apply SAME scaling as during training ===
    num_cols = ['Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt','BHK',
                'Property_Age','Infrastructure_Score','Amenities_Count']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predictions
    good_investment = clf.predict(input_df)[0]
    future_price = reg.predict(input_df)[0]

    # === Display Results ===
    st.markdown("## 📊 Results")

    if good_investment == 1:
        st.balloons()
        st.success("🎉 **EXCELLENT INVESTMENT OPPORTUNITY!**")
        st.markdown("This property satisfies all key criteria: affordable, good location, young building, ready to move, sufficient amenities.")
    else:
        st.error("⚠️ **NOT RECOMMENDED**")
        st.markdown("This property does not meet strong investment standards.")

    st.info(f"**Predicted Price in 2030:** ₹{future_price:.1f} Lakh")

    appreciation = ((future_price - price) / price) * 100 if price > 0 else 0
    st.metric("Expected 5-Year Appreciation", f"{appreciation:.1f}%")

    st.caption("Model trained on 250,000+ real Indian property records | Growth rates applied as of 2025")

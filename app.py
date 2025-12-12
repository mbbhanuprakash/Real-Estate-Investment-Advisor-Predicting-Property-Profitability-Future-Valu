import streamlit as st
import joblib
import pandas as pd

# Set page config for a clean, centered look
st.set_page_config(page_title="Real Estate Investment Advisor", layout="centered")

# Title
st.title("Real Estate Investment Advisor")

# Load pre-trained models
clf = joblib.load('classifier.pkl')
reg = joblib.load('regressor.pkl')

# Input fields in two columns for better layout
col1, col2 = st.columns(2)

with col1:
    size = st.number_input("Size (SqFt)", min_value=500, max_value=5000, value=1200)
    price = st.number_input("Price (₹ Lakhs)", min_value=20.0, max_value=1000.0, value=100.0)
    bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5])

with col2:
    city = st.selectbox("City", options=["Mumbai", "Bangalore", "Delhi", "Pune", "Hyderabad", "Other_City"])
    age = st.slider("Property Age (Years)", min_value=0, max_value=40, value=10)
    infra = st.slider("Infrastructure Score", min_value=3, max_value=15, value=10)

# Prediction button
if st.button("Predict"):
    # Calculate Price per SqFt
    ppsqft = (price * 100000) / size

    # Create input data frame matching your model's features
    data = pd.DataFrame([[size, price, ppsqft, bhk, age, infra, 5]],  # 5 is a placeholder for Amenities_Count
                        columns=['Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'BHK',
                                 'Property_Age', 'Infrastructure_Score', 'Amenities_Count'])

    # Add dummy columns for categorical variables (e.g., City) if needed
    for col in X.columns:  # X should be defined from your notebook
        if col not in data.columns:
            data[col] = 0
    data = data[X.columns]  # Ensure order matches training data

    # Make predictions
    investment_pred = clf.predict(data)[0]
    future_price = reg.predict(data)[0]

    # Display results
    if investment_pred == 1:
        st.balloons()  # Balloons for good investment
        st.success("EXCELLENT INVESTMENT OPPORTUNITY!")
    else:
        st.error("NOT RECOMMENDED FOR INVESTMENT")

    st.info(f"Predicted Price in 2030: ₹{future_price:.1f} Lakh")

# Note: Ensure 'X' (feature names) is available from your notebook's data preprocessing

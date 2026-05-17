import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("churn_model.pkl")

st.title("Spotify Subscription Churn Predictor")

st.write("Enter user behavior metrics to predict churn risk.")

# Inputs
listening_hours = st.slider("Listening hours per week", 0.0, 40.0, 10.0)
skip_ratio = st.slider("Skipped song ratio", 0.0, 1.0, 0.3)
playlist_created = st.slider("Playlists created", 0, 50, 5)
days_since_login = st.slider("Days since last login", 0, 60, 5)
premium_days = st.slider("Premium subscription duration (days)", 0, 1000, 100)
liked_songs = st.slider("Liked songs", 0, 1000, 100)
podcast_choice = st.selectbox("Uses podcasts?", ["No", "Yes"])
podcast_usage = 1 if podcast_choice == "Yes" else 0
social_shares = st.slider("Social shares per month", 0, 30, 2)

if st.button("Predict churn risk"):

    user_data = pd.DataFrame([{
        "listening_hours": listening_hours,
        "skip_ratio": skip_ratio,
        "playlist_created": playlist_created,
        "days_since_login": days_since_login,
        "premium_days": premium_days,
        "liked_songs": liked_songs,
        "podcast_usage": podcast_usage,
        "social_shares": social_shares
    }])

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High churn risk detected ({probability:.2%})")
        st.write("Recommended action: send discount or retention offer.")
    else:
        st.success(f"Low churn risk ({probability:.2%})")
        st.write("User likely to remain subscribed.")
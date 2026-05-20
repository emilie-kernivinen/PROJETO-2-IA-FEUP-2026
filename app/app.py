import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Spotify Churn Predictor",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
    <style>
    /* Main app layout background (Deep Charcoal/Black) */
    .stApp {
        background-color: #0c0f0d; 
        color: #E2E8F0; 
    }
    
    /* Headers get a Bright Emerald Green */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Subtle divider lines (Muted Dark Green) */
    hr {
        border-color: #1c2820 !important;
    }

    /* Emerald-to-Olive Gradient Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1db954, #14803a) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important; 
        padding: 12px 30px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.25);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.4);
        background: linear-gradient(135deg, #22ca5d, #1db954) !important;
    }

    [data-testid="stMetricValue"] {
        color: #14803a !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    /* Metric sub-labels stylized in Sage/Moss Green */
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important; 
        font-size: 14px !important;
        opacity: 0.85;
    }

    /* Sidebar styled in Dark Forest Green background */
    section[data-testid="stSidebar"] {
        background-color: #0f1912 !important;
        border-right: 1px solid #1c2820;
    }
    
    div[data-baseweb="slider"] > div > div > div {
        background: #14803a !important;
    }
    
    div[role="slider"] {
        background-color: #14803a !important;
        border: 2px solid #14803a !important;
        box-shadow: 0px 0px 4px rgba(29, 185, 84, 0.6) !important;
    }

    div.stSlider > div > div > div {
        color: #14803a !important;
    }        
            
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #14803a !important; 
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("../models/best_churn_model.pkl")

# Title
st.title("Spotify Subscription Churn Predictor")
st.markdown("### AI-powered customer retention analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("User Behavior Inputs")

listening_hours = st.sidebar.slider(
    "Listening hours per week",
    0.0, 40.0, 10.0
)

skip_ratio = st.sidebar.slider(
    "Skipped song ratio",
    0.0, 1.0, 0.3
)

playlist_created = st.sidebar.slider(
    "Playlists created",
    0, 50, 5
)

days_since_login = st.sidebar.slider(
    "Days since last login",
    0, 60, 5
)

premium_days = st.sidebar.slider(
    "Premium subscription duration (days)",
    0, 1000, 100
)

liked_songs = st.sidebar.slider(
    "Liked songs",
    0, 1000, 100
)

podcast_choice = st.sidebar.selectbox(
    "Uses podcasts?",
    ["No", "Yes"]
)

podcast_usage = 1 if podcast_choice == "Yes" else 0

social_shares = st.sidebar.slider(
    "Social shares per month",
    0, 30, 2
)

# Create columns
sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

with sum_col1:
    st.metric(label="Listening Hours", value=f"{listening_hours} h/wk")
    st.metric(label="Days Since Login", value=f"{days_since_login} days")

with sum_col2:
    st.metric(label="Skip Ratio", value=f"{skip_ratio:.2f}")
    st.metric(label="Premium Duration", value=f"{premium_days} days")

with sum_col3:
    st.metric(label="Playlists Created", value=playlist_created)
    st.metric(label="Liked Songs", value=liked_songs)

with sum_col4:
    st.metric(label="Podcast Usage", value=podcast_choice)
    st.metric(label="Social Shares", value=f"{social_shares}/mo")

st.markdown("---")

# Prediction button
if st.button("Predict Churn Risk"):

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

    st.markdown("---")
    st.subheader("Prediction Results")

    # Probability bar
    st.progress(float(probability))

    # Risk level
    if probability < 0.35:
        risk_level = "Low"
        st.success(f"Low churn risk ({probability:.2%})")

    elif probability < 0.70:
        risk_level = "Medium"
        st.warning(f"Medium churn risk ({probability:.2%})")

    else:
        risk_level = "High"
        st.error(f"High churn risk ({probability:.2%})")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Churn Probability", f"{probability:.2%}")
    metric2.metric("Risk Level", risk_level)
    metric3.metric("Prediction", "Churn" if prediction == 1 else "Stay")

    st.markdown("---")

    # Recommendations
    st.subheader("Recommended Actions")

    if probability >= 0.70:

        st.write("""
        - Recommend business plan
        - Send retention email campaign
        - Suggest personalized playlists
        """)

    elif probability >= 0.35:

        st.write("""
        - Increase engagement notifications
        - Recommend podcasts/playlists
        - Promote new features
        """)

    else:

        st.write("""
        - Maintain current engagement strategy
        - Recommend premium features
        """)

    st.markdown("---")
    st.subheader("Behavioral Insights")

    insights = []

    if days_since_login > 20:
        insights.append("High inactivity detected.")

    if listening_hours < 5:
        insights.append("Low listening engagement.")

    if skip_ratio > 0.6:
        insights.append("High song skipping behavior.")

    if premium_days > 500:
        insights.append("Long-term subscriber loyalty detected.")

    if playlist_created > 10:
        insights.append("Strong playlist engagement.")

    if len(insights) == 0:
        insights.append("No major behavioral risk factors detected.")

    for insight in insights:
        st.write(f"- {insight}")
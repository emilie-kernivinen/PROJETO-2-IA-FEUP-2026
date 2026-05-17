import numpy as np
import pandas as pd

n = 5000 

listening_hours = np.clip(np.random.normal(12, 6, n), 0, 40)
skip_ratio = np.clip(np.random.beta(2, 5, n), 0, 1)
playlist_created = np.random.poisson(8, n)
days_since_login = np.random.randint(0, 60, n)
premium_days = np.random.randint(0, 1000, n)
liked_songs = np.random.poisson(200, n)
podcast_usage = np.random.randint(0, 2, n)
social_shares = np.random.poisson(3, n)

risk_score = (
    - 0.15 * listening_hours
    + 3.0 * skip_ratio
    - 0.03 * playlist_created
    + 0.08 * days_since_login
    - 0.002 * premium_days
    - 0.002 * liked_songs
    - 0.5 * podcast_usage
    - 0.1 * social_shares
)

churn_probability = 1 / (1 + np.exp(-risk_score))
churn = np.random.binomial(1, churn_probability)

data = pd.DataFrame({
    "listening_hours": listening_hours,     
    "skip_ratio": skip_ratio,
    "playlist_created": playlist_created,
    "days_since_login": days_since_login,
    "premium_days": premium_days,
    "liked_songs": liked_songs,
    "podcast_usage": podcast_usage,
    "social_shares": social_shares,
    "churn": churn
})

data.to_csv("synthetic_spotify_users.csv", index=False)
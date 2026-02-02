import joblib
import numpy as np

# Load model and scaler
model = joblib.load("instagram_fake_account_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input (change values if needed)
# Order must match training features
sample_account = np.array([[  
    1,      # has_profile_pic
    50,     # bio_length
    0.3,    # username_randomness
    120,    # followers
    300,    # following
    0.4,    # follower_following_ratio
    200,    # account_age_days
    10,     # posts
    0.05,   # posts_per_day
    0.2,    # caption_similarity_score
    0.3,    # content_similarity_score
    0.1,    # follow_unfollow_rate
    0.05,   # spam_comments_rate
    0.1,    # generic_comment_rate
    0,      # suspicious_links_in_bio
    0       # verified
]])

# Scale input
sample_scaled = scaler.transform(sample_account)

# Predict
prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("🚨 Fake / Spam Instagram Account")
else:
    print("✅ Genuine Instagram Account")

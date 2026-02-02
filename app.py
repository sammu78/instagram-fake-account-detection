from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("instagram_fake_account_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        # Read input values from form
        data = {
            "has_profile_pic": int(request.form["has_profile_pic"]),
            "bio_length": float(request.form["bio_length"]),
            "username_randomness": float(request.form["username_randomness"]),
            "followers": float(request.form["followers"]),
            "following": float(request.form["following"]),
            "follower_following_ratio": float(request.form["follower_following_ratio"]),
            "account_age_days": float(request.form["account_age_days"]),
            "posts": float(request.form["posts"]),
            "posts_per_day": float(request.form["posts_per_day"]),
            "caption_similarity_score": float(request.form["caption_similarity_score"]),
            "content_similarity_score": float(request.form["content_similarity_score"]),
            "follow_unfollow_rate": float(request.form["follow_unfollow_rate"]),
            "spam_comments_rate": float(request.form["spam_comments_rate"]),
            "generic_comment_rate": float(request.form["generic_comment_rate"]),
            "suspicious_links_in_bio": int(request.form["suspicious_links_in_bio"]),
            "verified": int(request.form["verified"])
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Scale input
        scaled_data = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            result = "🚨 Fake / Spam Instagram Account"
        else:
            result = "✅ Genuine Instagram Account"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

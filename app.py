from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load trained model and scaler
try:
    model = joblib.load(os.getenv("MODEL_PATH", "instagram_fake_account_model.pkl"))
    scaler = joblib.load(os.getenv("SCALER_PATH", "scaler.pkl"))
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        try:
            # Explicitly define feature order to stay consistent with training
            features = [
                "has_profile_pic", "bio_length", "username_randomness", "followers",
                "following", "follower_following_ratio", "account_age_days", "posts",
                "posts_per_day", "caption_similarity_score", "content_similarity_score",
                "follow_unfollow_rate", "spam_comments_rate", "generic_comment_rate",
                "suspicious_links_in_bio", "verified"
            ]
            
            # Read input values from form with validation
            data = {}
            for feature in features:
                val = request.form.get(feature)
                if val is None or val.strip() == "":
                    raise ValueError(f"Missing value for {feature}")
                data[feature] = float(val) if "rate" in feature or "score" in feature or "randomness" in feature or "ratio" in feature or "per_day" in feature else int(val)

            if not model or not scaler:
                result = "❌ Error: Model or Scaler not loaded."
            else:
                # Convert to DataFrame
                df = pd.DataFrame([data])
                df = df[features] # Ensure correct order

                # Scale input
                scaled_data = scaler.transform(df)

                # Predict
                prediction = model.predict(scaled_data)

                if prediction[0] == 1:
                    result = "🚨 Fake / Spam Instagram Account"
                else:
                    result = "✅ Genuine Instagram Account"
        except Exception as e:
            result = f"❌ Error: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

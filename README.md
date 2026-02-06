# Instagram Fake Account Detection

A machine learning-powered web application to identify potential fake or spam Instagram accounts based on various profile features.

## 🚀 Features
- **Machine Learning**: Uses a Logistic Regression model trained on social media behavioral data.
- **Modern UI**: Dark-themed, responsive web interface with glassmorphism aesthetics.
- **Real-time Prediction**: Instant feedback on account genuineness.
- **Robustness**: Input validation and environment-based configuration.

## 🛠️ Project Structure
- `app.py`: Flask web server with prediction logic.
- `train_model.py`: Script to train the Logistic Regression model using `fake_social_media.csv`.
- `templates/index.html`: Modernized frontend.
- `.env`: Environment configuration for ports and debug mode.
- `requirements.txt`: Project dependencies.

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Re-train the model:
   ```bash
   python train_model.py
   ```

### Running the App
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000` (or the port specified in `.env`).

## 📊 Features Used for Prediction
The model analyzes 16 key indicators:
- Profile picture presence
- Bio length and content
- Username complexity (randomness)
- Follower/Following counts and ratios
- Account age
- Posting frequency
- Comment behavior (spam/generic rates)
- Verified status

---
&copy; 2026 InstaVerify

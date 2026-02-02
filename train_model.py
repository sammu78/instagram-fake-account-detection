import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
df = pd.read_csv("fake_social_media.csv")

print("Dataset loaded successfully")
print("Columns:", df.columns)

# -----------------------------
# STEP 2: Filter Instagram Data
# -----------------------------
df = df[df["platform"].str.lower() == "instagram"]

print("Instagram records:", len(df))

# -----------------------------
# STEP 3: Features & Target
# -----------------------------
X = df.drop(columns=["is_fake", "platform"])
y = df["is_fake"]

# -----------------------------
# STEP 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# STEP 6: Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# STEP 7: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: Save Model & Scaler
# -----------------------------
joblib.dump(model, "instagram_fake_account_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import time

# Load the dataset
df = pd.read_csv("synthetic_typing_sentiment_data.csv")

# Preprocess the features and labels
X = df[["typing_speed", "error_rate", "sentiment_score"]]
y = df["mood_label"]

# Encode mood labels into numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y_encoded)

# --- Streamlit UI ---
# Title and instructions
st.title("🧠 Mood Prediction Based on Typing")
st.write("Type something below. The app will predict your mood based on typing speed, sentiment, and errors.")

# User input
user_text = st.text_area("✍️ Start typing:", "")

# Start time for typing (saved between interactions)
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# Button to trigger prediction
if st.button("🔮 Predict Mood"):
    end_time = time.time()
    total_time = end_time - st.session_state.start_time

    # Features:
    typing_speed = round(len(user_text) / total_time, 2) if total_time > 0 else 0.1
    error_rate = 1.5  # Placeholder — can be improved
    sentiment_score = TextBlob(user_text).sentiment.polarity

    # Show features
    st.write(f"📏 Typing Speed: `{typing_speed}` chars/sec")
    st.write(f"❌ Error Rate: `{error_rate}` (simulated)")
    st.write(f"🗣️ Sentiment Score: `{sentiment_score}`")

    # Prepare for prediction
    input_data = pd.DataFrame([[typing_speed, error_rate, sentiment_score]],
                              columns=["typing_speed", "error_rate", "sentiment_score"])
    
    prediction = clf.predict(input_data)
    predicted_mood = le.inverse_transform(prediction)[0]

    st.success(f"🎉 Your predicted mood is: **{predicted_mood.upper()}**")

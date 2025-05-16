
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Text cleaning
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Sample training data
data = {
    'text': [
        "I am so happy today!",
        "This is the worst day ever.",
        "I feel amazing and joyful.",
        "Why does this always happen to me?",
        "I can't stop crying.",
        "I am furious about what happened!",
        "It's a wonderful morning.",
        "I am scared of the dark.",
        "You make me so angry.",
        "Life is beautiful."
    ],
    'emotion': [
        "happy", "sad", "happy", "sad", "sad", "angry",
        "happy", "fear", "angry", "happy"
    ]
}

# Prepare data
df = pd.DataFrame(data)
df['clean_text'] = df['text'].apply(clean_text)

# Build and train pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
pipe.fit(df['clean_text'], df['emotion'])

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🧠 Emotion Detection from Text")
st.write("Enter a sentence to detect the emotion.")

user_input = st.text_area("Your Text", "I feel great today!")

if st.button("Detect Emotion"):
    cleaned = clean_text(user_input)
    prediction = pipe.predict([cleaned])[0]
    st.success(f"Predicted Emotion: **{prediction.upper()}**")

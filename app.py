import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import re

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('product_reviews.csv')
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            st.error("Dataset must contain 'review' and 'sentiment' columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Custom text preprocessing function
def clean_text(text):
    try:
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        return text
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

# Preprocess data
def preprocess_data(df):
    try:
        df = df.dropna(subset=['review', 'sentiment'])
        df['review'] = df['review'].apply(clean_text)
        X = df['review']
        y = df['sentiment']
        
        # Encode the 'sentiment' column
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(X)
        
        # Split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)
        
        # Check if we have enough samples to apply SMOTE
        if X_train.shape[0] <= 1:
            st.warning("Not enough data to apply SMOTE. Consider using a larger dataset.")
            return X_train, X_test, y_train, y_test, vectorizer, le
        
        # Handle class imbalance using SMOTE with adjusted k_neighbors
        k_neighbors = min(5, X_train.shape[0] - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        return X_train_balanced, X_test, y_train_balanced, y_test, vectorizer, le
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None, None, None, None, None

# Train the model
def train_model(X_train, y_train):
    try:
        model = LogisticRegression(C=0.5)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

# Predict sentiment
def predict(model, vectorizer, le, input_text):
    try:
        input_text = clean_text(input_text)
        input_tfidf = vectorizer.transform([input_text])
        prediction = model.predict(input_tfidf)
        prediction_label = le.inverse_transform(prediction)
        return prediction_label[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return "Error"

# Streamlit UI
st.title('Product Review Sentiment Analysis')

df = load_data()
if df is not None:
    if st.checkbox('Show data sample'):
        st.write(df.head())

    X_train, X_test, y_train, y_test, vectorizer, le = preprocess_data(df)
    if X_train is not None and vectorizer is not None:
        model = train_model(X_train, y_train)
        if model is not None:
            user_input = st.text_area("Enter a product review to analyze its sentiment:")

            if user_input.strip():
                prediction = predict(model, vectorizer, le, user_input)
                if prediction == "Error":
                    st.error("There was an error processing your request.")
                elif prediction == 'positive':
                    st.success("The sentiment of the review is Positive!")
                else:
                    st.error("The sentiment of the review is Negative!")
            else:
                st.warning("Please enter a valid review.")

            if st.checkbox('Evaluate model'):
                try:
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
                    st.text('Classification Report:')
                    st.text(report)
                except Exception as e:
                    st.error(f"Error evaluating model: {e}")

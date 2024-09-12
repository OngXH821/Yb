import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('product_reviews.csv')
    
    # Check for missing data
    if df.isnull().values.any():
        st.write("Data contains missing values. Handling missing data...")
        df = df.dropna()  # Dropping missing data for simplicity
    
    return df

# Preprocess data
def preprocess_data(df):
    X = df['review']
    y = df['sentiment']

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Handle data imbalance using SMOTE, reducing k_neighbors to 1 for small datasets
    smote = SMOTE(random_state=42, k_neighbors=1)  # Use 1 neighbor to avoid the error
    X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

    return X_train_tfidf_resampled, X_test_tfidf, y_train_resampled, y_test, vectorizer

# Train the model
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

# Predict sentiment
def predict(model, vectorizer, input_text):
    try:
        input_tfidf = vectorizer.transform([input_text])
        prediction = model.predict(input_tfidf)
        
        # Log the prediction for debugging
        logging.debug(f"Prediction: {prediction[0]}")
        
        return prediction[0]
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit UI
st.title('Product Review Sentiment Analysis')

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Show data sample
if st.checkbox('Show data sample'):
    st.write(df.head())

# Preprocess data and train model
try:
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
    model = train_model(X_train_tfidf, y_train)
except Exception as e:
    st.error(f"Error during data preprocessing or model training: {e}")
    st.stop()

# Get user input
user_input = st.text_area("Enter a product review to analyze its sentiment:")

if user_input:
    prediction = predict(model, vectorizer, user_input)
    if prediction == 'positive':
        st.success("The sentiment of the review is Positive!")
    elif prediction == 'negative':
        st.error("The sentiment of the review is Negative!")
    else:
        st.error("Prediction error: unable to classify the input.")

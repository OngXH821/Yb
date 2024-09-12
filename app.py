import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('product_reviews.csv')
    return df

# Preprocess data
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    X = df['review']
    y = df['sentiment']

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Handle data imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

    return X_train_tfidf_resampled, X_test_tfidf, y_train_resampled, y_test, vectorizer


# Train the model
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

# Predict sentiment
def predict(model, vectorizer, input_text):
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    return prediction[0]

# Streamlit UI
st.title('Product Review Sentiment Analysis')

# Load data
df = load_data()

# Show data sample
if st.checkbox('Show data sample'):
    st.write(df.head())

# Preprocess data and train model
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
model = train_model(X_train_tfidf, y_train)

# Get user input
user_input = st.text_area("Enter a product review to analyze its sentiment:")

if user_input:
    prediction = predict(model, vectorizer, user_input)
    if prediction == 'positive':
        st.success("The sentiment of the review is Positive!")
    else:
        st.error("The sentiment of the review is Negative!")

# Evaluate the model
if st.checkbox('Evaluate model'):
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    st.text('Classification Report:')
    st.text(report)

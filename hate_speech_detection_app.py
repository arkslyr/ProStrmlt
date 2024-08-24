import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Title
st.title('Hate Speech Detection App')

# Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a classifier:", ["Decision Tree", "Random Forest"])

# Load default data
@st.cache_data
def load_data():
    df = pd.read_csv("HateSpeechDatasetBalanced.csv")
    return df

df = load_data()

# Data overview
if st.checkbox("Show data"):
    st.write(df.head())

# Preprocess data
X, y = df['Content'], df['Label']

# Split data
train_split = int(0.8 * X.shape[0])
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
@st.cache_resource
def train_models():
    classifier1 = DecisionTreeClassifier()
    classifier1.fit(X_train_vec, y_train)
    
    classifier2 = RandomForestClassifier()
    classifier2.fit(X_train_vec, y_train)
    
    return classifier1, classifier2

classifier1, classifier2 = train_models()

# Show Word Cloud
if st.checkbox("Show Word Cloud"):
    text1 = ' '.join(X_train.astype(str).tolist())
    wordcloud = WordCloud().generate(text1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Features")
    st.pyplot(plt)

# Data Upload
st.sidebar.title("Upload New Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write(new_data.head())
    
    X_new = new_data['Content']
    X_new_vec = vectorizer.transform(X_new)
    
    if options == "Decision Tree":
        predictions = classifier1.predict(X_new_vec)
    else:
        predictions = classifier2.predict(X_new_vec)
    
    st.write("Predictions for the uploaded data:")
    st.write(predictions)

# Input for prediction
user_input = st.text_area("Enter text for prediction:")

if user_input:
    user_input_vec = vectorizer.transform([user_input])
    
    if options == "Decision Tree":
        prediction = classifier1.predict(user_input_vec)
    else:
        prediction = classifier2.predict(user_input_vec)
    
    # Display prediction
    if prediction[0] == 0:
        st.write("The text is not hate speech.")
    else:
        st.write("The text is hate speech.")
        
# Show model accuracy
if st.checkbox("Show model accuracy"):
    if options == "Decision Tree":
        accuracy1 = accuracy_score(y_test, classifier1.predict(X_test_vec))
        st.write(f"Decision Tree Classifier Accuracy: {accuracy1 * 100:.2f}%")
    else:
        accuracy2 = accuracy_score(y_test, classifier2.predict(X_test_vec))
        st.write(f"RandomForest Classifier Accuracy: {accuracy2 * 100:.2f}%")

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os # Import os for path configuration

# --- NLTK Data Management ---
# Define a custom NLTK data path that is likely persistent in deployment environments.
# For Streamlit Cloud, the default /app/nltk_data is often a good choice.
# For local development, it will create 'nltk_data' in the current directory.
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

# Use st.cache_resource to download NLTK data only once per app deployment.
# This prevents repeated downloads on every app rerun, improving performance and reliability.
@st.cache_resource
def download_nltk_data():
    try:
        # Removed 'punkt_tab' as it's not a standalone downloadable package.
        # 'punkt' is the correct package for tokenization, and its integrity is key.
        # 'omw-1.4' is a dependency for 'wordnet' in newer NLTK versions.
        st.info(f"Checking NLTK data in: {NLTK_DATA_PATH}")
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('wordnet', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('omw-1.4', download_dir=NLTK_DATA_PATH, quiet=True)
        st.success("NLTK data ensured successfully!")
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}. Please check your internet connection, permissions, or NLTK data path.")
        st.stop() # Stop the app if essential data cannot be downloaded

download_nltk_data() # Execute the download function

# Initialize PorterStemmer after NLTK data is ensured
ps = PorterStemmer()

# --- Text Transformation Function ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text) # This is where 'punkt' is utilized

    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Ensure stopwords are loaded before use.
    # It's good practice to load this once if possible, but here it's fine.
    stop_words = set(stopwords.words('english'))
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Model and Vectorizer Loading ---
# Use st.cache_resource to load models only once per app deployment.
# This caches the loaded objects in memory, preventing repeated disk I/O.
@st.cache_resource
def load_resources():
    try:
        # Load the trained vectorizer and model
        # Ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory as the script
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory as the app.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"Failed to load model resources: {e}")
        st.stop()

tfidf, model = load_resources() # Execute the model loading function

# --- Streamlit UI ---
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
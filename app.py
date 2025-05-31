import streamlit as st
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import joblib

# 1. NLTK Resource Setup: ensure required corpora are in a local nltk_data directory
nltk_data_dir = os.path.join(os.getcwd(), "resources", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)  # add local directory to NLTK data path

# Download needed NLTK resources if not already present (avoids repeated downloads)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', download_dir=nltk_data_dir)

# 2. Load the pre-trained model and TF-IDF vectorizer
#    Assumes 'model.pkl' and 'vectorizer.pkl' are in the same directory as this script
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 3. Define text preprocessing function using NLTK (lowercasing, tokenizing, stopwords removal, stemming)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Lowercase, tokenize, remove non-alphabetic tokens,
    filter out stopwords, and apply Porter stemming.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        # Keep alphabetic tokens only and remove stopwords
        if token.isalpha() and token not in stop_words:
            cleaned_tokens.append(stemmer.stem(token))
    return " ".join(cleaned_tokens)


# 4. Streamlit App Interface
st.set_page_config(page_title="Spam Classifier")
st.title("Spam Classification App")
st.write("Enter a message below to check if it is **spam** or **not spam**:")

input_text = st.text_area("Message:", height=150)

if st.button("Classify"):
    if not input_text.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess the input and make prediction
        processed = preprocess_text(input_text)
        vector_input = vectorizer.transform([processed])
        prediction = model.predict(vector_input)[0]

        # Display result (use colored messages for clarity)
        if prediction == 1:
            st.error("This message is **SPAM**.")
        else:
            st.success("This message is **NOT SPAM**.")

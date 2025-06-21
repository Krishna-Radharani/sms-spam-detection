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
resources_to_download = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4')
]

for resource_path, resource_name in resources_to_download:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            st.error(f"Failed to download {resource_name}: {e}")

# Alternative fallback: Download all at once if individual downloads fail
try:
    # Test if punkt_tab is available
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # If not found, download all required resources
    for _, resource_name in resources_to_download:
        nltk.download(resource_name, quiet=True)

# 2. Load the pre-trained model and TF-IDF vectorizer
#    Assumes 'model.pkl' and 'vectorizer.pkl' are in the same directory as this script
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()

# 3. Define text preprocessing function using NLTK (lowercasing, tokenizing, stopwords removal, stemming)
stemmer = PorterStemmer()

# Initialize stopwords with error handling
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    # Fallback if stopwords not available
    stop_words = set()
    st.warning("Stopwords not available. Proceeding without stopword removal.")


def preprocess_text(text):
    """
    Lowercase, tokenize, remove non-alphabetic tokens,
    filter out stopwords, and apply Porter stemming.
    """
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned_tokens = []
        for token in tokens:
            # Keep alphabetic tokens only and remove stopwords
            if token.isalpha() and token not in stop_words:
                cleaned_tokens.append(stemmer.stem(token))
        return " ".join(cleaned_tokens)
    except Exception as e:
        st.error(f"Error in text preprocessing: {e}")
        return text.lower()  # Return basic preprocessing as fallback


# 4. Streamlit App Interface
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam Classification App")
st.write("Enter a message below to check if it is **spam** or **not spam**:")

# Add some styling
st.markdown("""
<style>
.stTextArea > div > div > textarea {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

input_text = st.text_area("Message:", height=150, placeholder="Enter your message here...")

# Add example messages
with st.expander("📝 Try these example messages"):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Example: Normal Message"):
            st.session_state.example_text = "Hi, how are you doing today? Let's meet for coffee."

    with col2:
        if st.button("Example: Spam Message"):
            st.session_state.example_text = "URGENT! You've won $1000! Click here now to claim your prize! Limited time offer!"

# Use example text if selected
if 'example_text' in st.session_state:
    input_text = st.session_state.example_text
    st.text_area("Message:", value=input_text, height=150, key="example_display")
    del st.session_state.example_text

if st.button("🔍 Classify Message", type="primary"):
    if not input_text.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                # Preprocess the input and make prediction
                processed = preprocess_text(input_text)

                # Check if preprocessing was successful
                if not processed.strip():
                    st.warning("⚠️ Message preprocessing resulted in empty text. Using original text.")
                    processed = input_text.lower()

                vector_input = vectorizer.transform([processed])
                prediction = model.predict(vector_input)[0]

                # Get prediction probability for confidence score
                try:
                    prediction_proba = model.predict_proba(vector_input)[0]
                    confidence = max(prediction_proba) * 100
                except:
                    confidence = None

                # Display result with confidence score
                st.markdown("---")
                if prediction == 1:
                    st.error("🚨 **This message is SPAM**")
                    if confidence:
                        st.error(f"Confidence: {confidence:.1f}%")
                else:
                    st.success("✅ **This message is NOT SPAM**")
                    if confidence:
                        st.success(f"Confidence: {confidence:.1f}%")

                # Show processed text for debugging (optional)
                with st.expander("🔧 View Processed Text (Debug Info)"):
                    st.text(f"Original: {input_text}")
                    st.text(f"Processed: {processed}")

        except Exception as e:
            st.error(f"❌ An error occurred during classification: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")

# Add footer information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>This app uses machine learning to classify messages as spam or not spam.</p>
    <p>Accuracy may vary depending on the message content and training data.</p>
</div>
""", unsafe_allow_html=True)

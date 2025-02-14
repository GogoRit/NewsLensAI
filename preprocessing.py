import re

def preprocess_text(text):
    """
    Clean and preprocess the input text.
    """
    # Remove extra spaces and special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s.,]', '', text)  # Remove special characters except punctuation
    return text.strip()
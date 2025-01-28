import pandas as pd
import spacy

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to load and clean the data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Ensure the necessary columns are present
    if 'title' not in df.columns or 'cleaned_article' not in df.columns:
        raise ValueError("CSV file must contain 'title' and 'cleaned_article' columns.")
    
    # Drop rows with missing articles
    df.dropna(subset=['cleaned_article'], inplace=True)
    df['cleaned_article'] = df['cleaned_article'].str.strip()  # Clean any whitespace
    
    return df

# Function for NER extraction
def ner_extraction_tool(article):
    doc = nlp(article)
    entities = []
    for ent in doc.ents:
        entities.append({'text': ent.text, 'label': ent.label_})
    return entities

# Function to preprocess the article text (removing stopwords, lemmatizing)
def preprocess_text(article):
    doc = nlp(article)
    cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text

# Main preprocessing function
def preprocess_articles(file_path):
    # Load and clean the data
    df = load_and_clean_data(file_path)
    
    # Apply text preprocessing (lemmatization, stopwords removal)
    df['cleaned_article'] = df['cleaned_article'].apply(preprocess_text)
    
    # Apply NER extraction to the cleaned articles
    df['ner_entities'] = df['cleaned_article'].apply(ner_extraction_tool)
    
    return df

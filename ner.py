import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from the input text using spaCy.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
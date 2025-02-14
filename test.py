import spacy

# Load the model
nlp = spacy.load("en_core_web_sm")

# Test the model
doc = nlp("This is a test sentence.")
print([(ent.text, ent.label_) for ent in doc.ents])
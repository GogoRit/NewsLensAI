import pytest
from ner.entity_extractor import extract_entities

def test_basic_extraction():
    text = "Apple Inc. is headquartered in Cupertino, California."
    ents = extract_entities(text, min_score=0.5)
    labels = {e.label for e in ents}
    texts  = {e.text for e in ents}

    assert "ORG" in labels
    assert "Apple Inc" in texts
    assert "Cupertino" in texts
    assert "California" in texts
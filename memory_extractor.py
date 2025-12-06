import spacy
import json
import os
from datetime import datetime

class MemoryStream:
    def __init__(self):
        print("--- Loading NER Model (SpaCy) ---")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading language model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.memory_db = []

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            # LABEL_ means "PERSON", "GPE" (Location), "DATE", etc.
            entities[ent.label_] = ent.text
        return entities

    def process_text(self, user_id, text):
        """
        Analyzes text for entities and saves to 'database' if found.
        """
        extracted_data = self.extract_entities(text)
        
        if extracted_data:
            record = {
                "user_id": user_id,
                "timestamp": str(datetime.now()),
                "raw_text": text,
                "extracted_info": extracted_data
            }
            self.memory_db.append(record)
            return f"[MEMORY UPDATED] {json.dumps(extracted_data)}"
        return None

if __name__ == "__main__":
    mem = MemoryStream()
    print(mem.process_text("u1", "My name is Anvesh and I am going to Delhi on Monday."))

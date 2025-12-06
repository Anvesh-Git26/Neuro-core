import torch
import os
import sys
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# FIX IMPORTS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from knowledge_engine import KnowledgeEngine

class MikoRobot:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # FIX PATH: Find the model relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '..', 'miko_model')

        if os.path.exists(model_path):
            self.classifier = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            print("Warning: Model not found. Using default untrained model.")
            self.classifier = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            
        self.classifier.to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.brain = KnowledgeEngine()

    def get_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        predicted_id = logits.argmax().item()
        return "KNOWLEDGE" if predicted_id == 1 else "ACTION"

    def process(self, text):
        print(f"\nUser: {text}")
        intent = self.get_intent(text)
        print(f"Intent: {intent}")
        
        if intent == "KNOWLEDGE":
            answer = self.brain.search(text)
            print(f"Miko: {answer}")
        else:
            print("Miko: [Executing Movement Sequence...]")

if __name__ == "__main__":
    bot = MikoRobot()
    bot.process("Miko dance for me")
    bot.process("Who invented the telephone?")

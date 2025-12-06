import torch
import os
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification

# FIX IMPORTS: Add the current directory to Python path so we can import local files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import MikoIntentDataset

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")

    # FIX PATHS: Make dataset path dynamic and robust
    # Get the directory where THIS script is located (src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to find data/ (../data/dataset.json)
    dataset_path = os.path.join(base_dir, '..', 'data', 'dataset.json')

    if not os.path.exists(dataset_path):
        print(f"Error: dataset.json not found at {dataset_path}")
        return

    dataset = MikoIntentDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("Starting Training...")
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # FIX SAVE PATH: Save model relative to this script
    save_path = os.path.join(base_dir, '..', 'miko_model')
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()

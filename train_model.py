import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification
# Note: If running from root, use: from src.data_loader import MikoIntentDataset
from data_loader import MikoIntentDataset

def train():
    # 1. Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")

    # 2. Data
    # Assuming we run this script from inside the src folder
    dataset_path = '../data/dataset.json' 
    if not os.path.exists(dataset_path):
        print("Error: dataset.json not found. Check path.")
        return

    dataset = MikoIntentDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)
    model.train()

    # 4. Loop
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

    # 5. Save
    save_path = "../miko_model"
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
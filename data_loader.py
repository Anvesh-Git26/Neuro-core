import json
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class MikoIntentDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.label_map = {"ACTION": 0, "KNOWLEDGE": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label_id = self.label_map[item['label']]
        
        encoding = tokenizer(
            text,
            padding='max_length', 
            truncation=True,
            max_length=16, 
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
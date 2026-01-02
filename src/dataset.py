import torch
import torch.utils.data import Dataset
import pandas as pd
from typing import Dict

class NepaliGECDataset(Dataset):
    
    def __init__(self, data_path:str,tokenizer,max_source_length:int = 128,max_target_length:int=128):
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        print(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        incorrect_text = str(row['incorrect'])
        correct_text = str(row['correct'])
        
        source_ids = self.tokenizer.encode(
            incorrect_text,
            add_bos = True,
            add_eos = True
        )
        
        target_ids = self.tokenizer.encode(
            correct_text,
            add_bos = True
            add_eos = True
        )
        
        return {
            "input_ids" : source_ids,
            "labels" : target_ids
        }
        

          
    
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
            add_eos = True,
            max_length=self.max_source_length,
            truncation=True
        )
        
        target_ids = self.tokenizer.encode(
            correct_text,
            add_bos = True,
            add_eos = True,
            max_length=self.max_target_length,
            truncation=True
        )
        
        return {
            "input_ids" : source_ids,
            "labels" : target_ids
        }
        
class Seq2SeqCollator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id
        
    def __call__(self, batch):
        #Encoder
        input_ids = [torch.tensor(b['input_ids'],dtype=torch.long) for b in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad)
        attention_mask = (input_ids != self.pad).long()
        
        #Decoder
        targets = [b['labels'] for b in batch]
        decoder_input_ids = [torch.tensor(t[:-1],dtype=torch.long) for t in targets]
        labels =  [torch.tensor(t[1:],dtype=torch.long) for t in targets]
        
        decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.pad)
        
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad)
        
        labels[labels == self.pad] = -100 # important for loss
        
        decoder_attention_mask = (decoder_input_ids != self.pad).long()
        
        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "decoder_input_ids" : decoder_input_ids,
            "decoder_attention_mask" : decoder_attention_mask,
            "labels" : labels
        }
    
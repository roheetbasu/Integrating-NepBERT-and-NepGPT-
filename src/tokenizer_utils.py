import os
import sentencepiece as spm
import pandas as pd
from typing import Optional, List

class NepaliSentencePieceTokenizer:
    
    def __init__(
        self,
        model_prefix: str = "Nepali_tokenizer",
        vocab_size: int = 32000,
        model_type = "unigram",
        character_coverage: float = 0.9995,
        pad_id: int = 0,
        unk_id: int = 0,
        bos_id: int = 0,
        eos_id: int = 0
    ):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        self.model_file = f"{model_prefix}.model"
        self.vocab_file = f"{model_prefix}.vocab"
        
        self.sp = None
        
    def load(self):
        
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(
                f"Tokenizer model not found from {self.model_file}"
                "please train the tokenzier first"
            )
            
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)
        
        print(f" Loaded tokenizer from {self.model_file}")
        print(f" Vocabulary size: {self.vocab_size}")
        
    def train(self, data_paths: List[str], force_retrain: bool = False):
        
        if os.path.exists(self.model_file) and not force_retrain:
            print(f"Tokenizer model found at {self.model_file}")
            print("Loading existing tokenzier.....")
            self.load()
            
        print("Training tokenizer....")
        
        all_texts = []
        
        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"Warning:{data_path} not found, skipping...")
                continue
            
            df = pd.read_csv(data_path)
            all_texts.extend(df.iloc[:,0].dropna().astype(str).tolist())
            
            if len(all_texts) == 0:
                raise ValueError(f'No training data found')
            
        #save to temporary files
        temp_file = f"{self.model_prefix}_temp_file.txt"
            
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text.strip() + '\n')
                
        print(f"Vocab_size:{self.vocab_size}")
        print(f"Model_type:{self.model_type}")
        print(f"Character_coverage:{self.character_coverage}")
        
        try:
            spm.SentencePieceProcessor(
                input = temp_file,
                model_prefix = self.model_prefix,
                character_coverage = self.character_coverage,
                pad_id = self.pad_id,
                unk_id = self.unk_id,
                bos_id = self.bos_id,
                eos_id = self.eos_id,
                user_defined_symbols = ['[PAD]','[UNK]','[BOS]','[EOS]'],
                normalization_rule_name = 'nfkc'
            )
            
            print(f"Tokenizer Trained Sucessfully.....")
            print(f"model saved to: {self.model_file}")
            print(f"Vocabulary saved to: {self.vocab_file}")
            
        finally:
            #clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
        self.load()                     

    def encode(self, text: str, add_bos: bool=False, add_eos: bool=False):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer Not loaded")
        
        ids = self.sp.encode(text, out_type = int)
        
        if add_bos:
            ids = [self.bos_id] + ids
            
        if add_eos:
            ids = ids + [self.eos_id]
            
        return ids
    
    def decode(self, ids: List[int]):
        
        if self.sp is None:
            raise RuntimeError("Tokenzier not loaded")
        
        return self.sp.decode(ids)
    
    def encode_batch(self, texts: List[str], add_bos:bool=False, add_eos:bool=False):
        
        return [self.encode(text, add_bos, add_eos) for text in texts]
    
    def decode_batch(self, ids_list:List[List[int]]):
        
        return [self.decode(ids) for ids in ids_list]
    
    def tokenize(self, text: str):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer Not loaded")
        
        return self.sp.encode(text, out_type = str)
            
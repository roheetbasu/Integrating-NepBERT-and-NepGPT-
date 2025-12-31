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
        
        if not os.path.exists()
        
    def train(self, data_paths: List[str], force_retrain: bool = False):
        
        if os.path.exists(self.model_file) and not force_retrain:
            print(f"Tokenizer model found at {self.model_file}")
            print("Loading existing tokenzier.....")
            self.load()
            
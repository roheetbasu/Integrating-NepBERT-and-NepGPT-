import os
import sentencepiece as spm
import numpy as np
from typing import Optional, List

class NepaliSentencePieceTokenizer:
    
    def __init__(
        self,
        model_prefix: str = "Nepali_tokenizer",
        vocab_size: int = 32000,
        model_type = "unigram",
        character_coverage: float = 0.9995,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
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
                "please train the tokenizer first"
            )
            
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)
        
        print(f" Loaded tokenizer from {self.model_file}")
        print(f" Vocabulary size: {self.sp.vocab_size}")
        
    def train(self, data_path: str, force_retrain: bool = False):
        
        if os.path.exists(self.model_file) and not force_retrain:
            print(f"Tokenizer model found at {self.model_file}")
            print("Loading existing tokenzier.....")
            self.load()
            return
            
        print("Training tokenizer....")
        
        all_texts = []
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")
            
            
        with open(data_path, "r", encoding="utf-8") as f:
            all_texts = [line.strip() for line in f if line.strip()]

            
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
            spm.SentencePieceTrainer.train(
                input = temp_file,
                model_prefix = self.model_prefix,
                vocab_size  = self.vocab_size,
                model_type = self.model_type,
                character_coverage = self.character_coverage,
                pad_id = self.pad_id,
                unk_id = self.unk_id,
                bos_id = self.bos_id,
                eos_id = self.eos_id,
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

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False, max_length: int = 128, truncation: bool = True):
        if self.sp is None:
            raise RuntimeError("Tokenizer Not loaded")
        
        ids = self.sp.encode(text, out_type = int, add_bos=False, add_eos=False)
        
        if add_bos:
            ids = [self.bos_id] + ids

        if max_length is not None and truncation:
            # Reserve space for EOS if needed
            reserve = 1 if add_eos else 0
            ids = ids[:max_length - reserve]

        if add_eos:
            ids = ids + [self.eos_id]
                
        return ids
    
    def decode(self, ids: List[int]):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.sp.decode(ids)
    
    def encode_batch(self, texts: List[str], add_bos=False, add_eos=False, max_length: int = 128):
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        
        ids_batch = self.sp.encode(texts, out_type=int)  # Batch encode all sentences
        
        processed_batch = []
        for ids in ids_batch:
            ids = np.array(ids, dtype=np.int32)
            
            # Reserve space for EOS if needed
            reserve = 1 if add_eos else 0
            
            # Truncate first
            if max_length is not None:
                ids = ids[: max_length - reserve]
            
            # Add BOS
            if add_bos:
                ids = np.insert(ids, 0, self.bos_id)
            
            # Add EOS
            if add_eos:
                ids = np.append(ids, self.eos_id)
            
            processed_batch.append(ids)
        
        return processed_batch  # Returns a list of NumPy arrays
                
    
    def decode_batch(self, ids_list:List[List[int]]):
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        
        return list(map(self.sp.decode, ids_list))
    
    def tokenize(self, text: str):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer Not loaded")
        
        return self.sp.encode(text, out_type = str)
    
    def get_vocab_size(self):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.sp.vocab_size
            
    def id_to_piece(self, idx : int):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.sp.id_to_piece(idx)
    
    def piece_to_id(self, piece:str):
        
        if self.sp is None:
            raise RuntimeError("Tokenizer no loaded")
        
        return self.sp.piece_to_id(piece)
    
    @property
    def pad_token_id(self):
        return self.pad_id
    
    @property
    def unk_token_id(self):
        return self.unk_id
    
    @property
    def bos_token_id(self):
        return self.bos_id
    
    @property
    def eos_token_id(self):
        return self.eos_id
    
    
        
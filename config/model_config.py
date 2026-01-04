from dataclasses import dataclass
from typing import Optional

@dataclass
class EncoderDecoderConfig:
    
    # Encoder setting
    encoder_model_name:str  = "Rajan/NepaliBERT"
    encoder_hidden_size:int = 768
    
    # Decoder setting
    decoder_model_name: str = "Rajan/NepGPT"
    decoder_hidden_size:int = 768 
    
    # Cross-attention settings
    num_cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1
    
    #General_setting
    max_source_length:int = 128
    max_target_length:int = 128
    vocab_size:int = 32000
    
    #Training setting
    batch_size: int = 16
    num_epochs: int = 10 
    learning_rate:float = 5e-5
    gradient_accumulation:int = 4
    
    # Generation settings
    num_beams: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True
    
    #tokenizer
    data_path = ""
    model_prefix: str = "Nepali_tokenizer"
    vocab_size: int = 32000
    model_type = "unigram"
    
    
    
    
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
from models.cross_attention import CrossAttentionLayer
from typing import  Optional, Dict

class NepaliBERTNepGPTModel(nn.Module):
    """
    Encoder Decoder model with NepaliBERT encoder and NepGPT decoder
    connected via cross attention
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #Load NepBERT
        print("Loading NepBERT...")
        self.encoder = AutoModel(config.encoder_model_name)
        
        #Load NepGPT
        print("Loading NepGPT...")
        self.decoder = AutoModelForCausalLM(config.decoder_model_name)
        
        #Add the cross-attention layer 
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size = config.decoder_hidden_size,
                num_heads = config.num_cross_attention_head,
                dropout = config.cross_attention_dropout
                                )
            for _ in range(self.decoder.config.num_hidden_layers)
        ])
        
        #layer norms for residual connection
        self.cross_attns_layer_norms = nn.ModuleList([
            nn.LayerNorm(config.decoder_hidden_size)
            for _ in range(self.deocder.config.num_hidden_layers)
        ])
        
        # Projection if encoder and decoder sizes differ
        if config.encoder_hidden_size != config.decoder_hidden_size:
            self.encoder_proj = nn.Linear(
                config.encoder_hidden_size, 
                config.decoder_hidden_size
            )
        else:
            self.encoder_proj = None
            
    
    def forward(self, inputs_ids)
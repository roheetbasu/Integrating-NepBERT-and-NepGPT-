import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossAttentionLayer(nn.Module):
    """_summary_
    cross attention layer to connect Encoder(NepBERT) and Decoder(NepGPT)
    """
    
    def __init__(self, hidden_size: int,num_heads: int, dropout:float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = nn.Dropout(dropout)
        
        assert(hidden_size % num_heads == 0)
        
        self.head_dim = hidden_size/num_heads
        
        #Query from Decoder, Key and Value from Encoder
        self.query_proj = nn.Linear(hidden_size,hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size,hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size,hidden_size, bias = False)
        self.out_proj = nn.Linear(hidden_size,hidden_size)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_hidden: torch.Tensor, encoder_attention_mask: Optional[torch.Tensor] = None):
        
        batch_size = decoder_hidden.size(0)
        
        #project to q,k and v 
        Q = self.query_proj(decoder_hidden)# (batch, tgt_len, hidden_size)
        K = self.key_proj(encoder_hidden)# (batch, src_len, hidden_size)
        V = self.value_proj(encoder_hidden)# (batch, src_len, hidden_size)
        
        #Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2) # (batch, tgt_len, hidden_size) -> (batch, num_heads, tgt_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        #compute attention score (batch, num_heads, tgt_seq, head_dim) @ (batch, num_heads, head_dim, src_seq) -> (batch, num_heads, tgt_seq, src_seq)
        attention_scores = torch.matmul(Q,K.transpose(-2,-1)) * self.scale
        
        #apply masking if provided
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                encoder_attention_mask == 0, float('-inf')
            )
        
        #softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        #applying attention to values(batch, num_heads, tgt_seq, head_dim)
        context = torch.matmul(attention_weights, V)
        
        #Reshape back
        context = context.transpose(1,2).contigious().view(batch_size, -1, self.hidden_size)
        
        #Final projection
        output = self.out_proj(context)
        
        return output, attention_weights
        
        
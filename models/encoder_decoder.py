import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
from models.cross_attention import CrossAttentionLayer
from src.tokenizer_utils import NepaliSentencePieceTokenizer
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
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name)
        encoder_vocab_size = self.config.vocab_size
        if self.encoder.get_input_embeddings().weight.size(0) != encoder_vocab_size:
            self.encoder.resize_token_embeddings(encoder_vocab_size)

        
        #Load NepGPT
        print("Loading NepGPT...")
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_model_name)
        
        # Resize embeddings to match your tokenizer vocab
        decoder_vocab_size = self.config.vocab_size  # from NepaliSentencePieceTokenizer
        if self.decoder.get_input_embeddings().weight.size(0) != decoder_vocab_size:
            self.decoder.resize_token_embeddings(decoder_vocab_size)
            print(f"Resized decoder embeddings to vocab size: {decoder_vocab_size}")
        
        #Add the cross-attention layer 
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size = config.decoder_hidden_size,
                num_heads = config.num_cross_attention_heads,
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
            
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                decoder_input_ids : torch.Tensor,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                label: Optional[torch.Tensor] = None,
                ):
        
        #Encode source text with NepaliBERT
        encoder_outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        if self.encoder_proj is not None:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
            
        #get decoder embedding
        decoder_embedding = self.decoder.model.embed_tokens(decoder_input_ids)
        
        #pass through the decoder layers
        hidden_states = decoder_embedding
        
        for i, (decoder_layer, cross_attn_layer, layer_norm) in enumerate(
            zip(self.decoder.model.layers,
                self.cross_attention_layers,
                self.cross_attns_layer_norms)
        ):
            #self attention in decoder
            layer_output = decoder_layer(hidden_states)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            
            #cross attention to encoder
            cross_attn_output = cross_attn_layer(
                decoder_hidden = hidden_states,
                encoder_hidden = encoder_hidden_states,
                encoder_attention_mask = attention_mask
                )
            
            #Residual Connection
            hidden_states = layer_norm(hidden_states + cross_attn_output)
            
            #Final language modeling head
            lm_logits = self.decoder.lm_head(hidden_states)
            
            loss = None
            if label is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1,lm_logits.size(-1)),label.view(-1))
                
        return {
            'loss': loss,
            'logits' : lm_logits,
            'encoder_hidden_states' : encoder_hidden_states
        }
        
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        **kwargs   
    ):
        #Encode
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        if self.encoder_proj is not None:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
            
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Start with BOS token
        bos_token_id = self.config.bos_token_id
        decoder_input_ids = torch.full(
            (batch_size,1),
            bos_token_id,
            dtype=torch.long,
            device=device
        ) 
        
        # greedy decoding
        generated_ids = []
        
        for _ in range(max_length):
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )
            
            next_token_logits = output['logits'][:,-1,:]
            next_token_id = torch.argmax(next_token_logits, dim = -1,keepdim=True)
            
            generated_ids.append(next_token_id)
            decoder_input_ids = torch.cat([decoder_input_ids,next_token_id],dim=1)
            
            #stop if EOS token
            eos_token_id = self.config.eos_token_id
            if eos_token_id and (next_token_id == eos_token_id).all():
                break
            
            if generated_ids:
                return torch.cat(generated_ids, dim=1)
            return torch.tensor([[]], device=device, dtype=torch.long)
        

class GECModel:
    
    def __init__(self, config):
        
        self.config = config
        
        #Load Tokenizer first
        self.tokenizer = NepaliSentencePieceTokenizer(
            model_prefix=config.model_prefix
        )
        self.tokenizer.load()
        
        # enforce alignment
        config.pad_token_id = self.tokenizer.pad_token_id
        config.unk_token_id = self.tokenizer.unk_token_id
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
        config.vocab_size = self.tokenizer.get_vocab_size()
        
        #Load model
        self.model = NepaliBERTNepGPTModel(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on: {self.device}")
        
    def correct(self, text):
        
        input_ids = self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
            max_length = self.config.max_source_length,
            truncation=True
        )
        
        input_ids = torch.tensor([input_ids],dtype=torch.long).to(self.device)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_target_length
            )

        return self.tokenizer.decode(generated_ids[0].tolist())

        

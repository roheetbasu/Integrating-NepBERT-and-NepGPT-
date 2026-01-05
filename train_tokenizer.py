from src.tokenizer_utils import NepaliSentencePieceTokenizer
from config.model_config import EncoderDecoderConfig

def main(config):
    
    tokenizer = NepaliSentencePieceTokenizer(
        model_prefix=config.model_prefix,
        vocab_size=config.tokenizer_vocab_size,
        model_type=config.model_type,
        pad_id=config.pad_id,
        unk_id=config.unk_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id
    )
    
    tokenizer.train(config.data_path,force_retrain=config.force_retrain)
    
    # Test tokenizer
    print("\n" + "="*60)
    print("Testing Tokenizer")
    print("="*60)
    
    test_sentences = [
        "म स्कूल जान्छु",
        "उनी खेलिरहेका छन्",
        "हामी घर जान्छौं"
    ]
    
    for sent in test_sentences:
        tokens = tokenizer.tokenize(sent)
        ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids)
        
        print(f"\nOriginal: {sent}")
        print(f"Tokens:   {tokens}")
        print(f"IDs:      {ids}")
        print(f"Decoded:  {decoded}")
    
if __name__ == '__main__':
    config = EncoderDecoderConfig()
    main(config)
    
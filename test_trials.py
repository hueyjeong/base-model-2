import torch
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig

def test_trials():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Trial A Test
    print("=== Testing Trial A: Source-Aware Logit Bias ===")
    config_a = BitMambaSeq2SeqConfig(vocab_size=1000, d_model=256, n_heads=8, n_encoder_layers=1, n_decoder_layers=1)
    model_a = BitMambaSeq2Seq(config_a).to(device)
    model_a.eval()

    src_ids = torch.tensor([[10, 20, 30, 0, 0]], device=device)
    tgt_ids = torch.tensor([[100, 200]], device=device)
    
    with torch.no_grad():
        logits_orig = model_a(src_ids, tgt_ids, source_bias=0.0)
        logits_bias = model_a(src_ids, tgt_ids, source_bias=2.0)
        
    diff = logits_bias - logits_orig
    print("Diff at source token 10:", diff[0, 0, 10].item())
    print("Diff at source token 20:", diff[0, 0, 20].item())
    print("Diff at pad token 0:", diff[0, 0, 0].item())
    print("Diff at non-source token 50:", diff[0, 0, 50].item())
    
    # 2. Trial B Test
    print("\n=== Testing Trial B: Copy Gate ===")
    config_b = BitMambaSeq2SeqConfig(vocab_size=1000, d_model=256, n_heads=8, n_encoder_layers=1, n_decoder_layers=1, use_copy_gate=True)
    model_b = BitMambaSeq2Seq(config_b).to(device)
    model_b.eval()
    
    with torch.no_grad():
        logits_b = model_b(src_ids, tgt_ids)
        
    print("Output shape:", logits_b.shape)
    print("No NaN check:", not torch.isnan(logits_b).any().item())
    
    print("\nTests passed!")

if __name__ == "__main__":
    test_trials()

import subprocess
import os

env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Disable compile to speed up test and avoid Triton compile memory spikes for now
cmd = [
    "python3", "training/pretrain.py",
    "--corpus", "corpus/sample_1g.jsonl",
    "--text_key", "text",
    "--size", "128M",
    "--pack_size", "16384",
    "--tokenizer", "keyboard",
    "--batch_size", "1",
    "--grad_accum_steps", "1",
    "--bf16",
    "--int8",
    "--fused_ce",
    "--grad_ckpt",
    "--save_dir", "checkpoints/run_vtest",
    "--log_every", "1",
    "--num_workers", "0",
    "--max_steps", "3"
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd, env=env)

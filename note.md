# NOTE

## 8M pretrain
python -u -m training.pretrain \
  --tokenizer mecab_bbpe \
  --size 8M --corpus corpus/sample_1g.jsonl --text_key text \
  --pack_size 4096 --batch_size 16 --grad_accum_steps 1 --num_workers 16 \
  --grad_ckpt --amp --fused_ce --max_chars 500000000 --lr 5e-4 \
  --warmup_steps 1000 --log_every 1 --save_every 5000 \
  --val_corpus corpus/val_50k.jsonl --val_every 200 --val_steps 20 \
  --save_dir checkpoints/8M_mecab_bbpe
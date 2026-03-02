# Analysis: Context Capacity of Linear Attention for Seq Length 4096

The user asked: "Can this layer size (d_model=288, n_heads=4, MQA) hold the full information of a sequence length of 4096?"

## 1. How Linear Attention Stores Information
In standard Softmax Attention, the model keeps all $N$ tokens in memory ($N \times d$ matrix) and looks back at them.
In **Linear Attention**, the entire sequence of length $N$ is compressed into a single **$KV$ Context Matrix** of size $d_{head} \times d_{head}$ per head. 

- $K^T \in \mathbb{R}^{d_{head} \times N}$
- $V \in \mathbb{R}^{N \times d_{head}}$
- $Context\_Matrix = K^T V \in \mathbb{R}^{d_{head} \times d_{head}}$

By doing this, the sequence length $N$ (e.g., 4096) is summed over and disappears. The total storage capacity depends **entirely on $d_{head}$**, not $N$.

## 2. Capacity of the Current Architecture
For the 8M model currently running:
- `d_model` = 288
- `n_heads` = 4 (based on `d_model // 64`)
- `d_head` = 288 / 4 = 72
- **MQA (Multi-Query Attention)**: K and V use `n_kv_heads` = 1.

So the actual Context Matrix size is:
`d_head` (72) $\times$ `d_head` (72) = **5,184 elements** (FP16/BF16).

### 🚨 Information Bottleneck Analysis
We are trying to compress **4,096 tokens** (each containing 72-128 bits of useful semantic information depending on the dimension) into a matrix of **5,184 values**.

**Is this enough?**
- **No.** From an information-theoretic perspective, compressing 4096 distinct vectors of size 72 into a $72 \times 72$ matrix using simple outer product summation ($K^T V$) will result in **catastrophic superposition (interference)**. 
- The matrix will become strictly rank-deficient (max rank 72). When the decoder tries to extract information using a query $Q$, the distinct features of tokens past the first ~100-200 tokens will be irretrievably blurred together.

## 3. Comparison with Mamba
Mamba (d_model=288, d_state=16) compresses information into a state vector of size $d\_inner \times d\_state \approx 576 \times 16 = 9,216$ elements.
Our Linear Attention (MQA, d_head=72) compresses into $72 \times 72 = 5,184$ elements. 

So currently, our Cross-Attention has **less state capacity than the Mamba block itself**.

## 4. How to Fix This?
To support long sequences like 4096 without forgetting, the Context Matrix must be larger. The size of the context matrix grows quadratically with `d_head` ($d_{head}^2$).

1.  **Stop using MQA for Linear Attention ($n\_kv\_heads > 1$)**
    - Multi-Query Attention forces all Q heads to share a single $72 \times 72$ context matrix.
    - If we switch back to **Multi-Head Attention (MHA)** or **Grouped-Query Attention (GQA)**, each head gets its own $72 \times 72$ matrix.
    - With 4 heads (MHA), capacity quadruples: $4 \times (72 \times 72) = \mathbf{20,736}$ elements.

2.  **Increase `d_head`** 
    - The capacity is heavily dependent on the head dimension. If we were using a larger model (e.g., 128M model with `d_model=768`), `d_head` would be naturally larger, but for the 8M model, we need to artificially widen the keys/values.
    - For example, standard Transformer $d_{head}$ is 64 or 128. $128 \times 128 = 16,384$ elements per head.

3.  **The "GLU" Expansion Trick (Similar to Gated Linear Attention)**
    - Project K and V to a higher dimension internally before multiplying, expanding the rank of the context matrix.

## 5. Proposed Code Change for `decoder.py`
We should allow `n_kv_heads` to be set to `n_heads` (MHA) by default instead of `1` (MQA) to maximize capacity, as parameter size is not our primary bottleneck here (it's information capacity).

```python
        # Linear Cross-Attention
        from model.linear_attention import LinearCrossAttention
        self.cross_attn = LinearCrossAttention(
            d_model=d_model,
            n_heads=d_model // 64,  # e.g., 4 for 8M model
            n_kv_heads=d_model // 64, # <-- CHANGE: Use MHA instead of MQA for max capacity
            dropout=dropout,
        )
```

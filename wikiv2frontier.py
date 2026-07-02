import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from collections import Counter
from datasets import load_dataset
import math
import sys
import time
import os
import pickle
sys.stdout.reconfigure(encoding="utf-8")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('device is: ', device)
# parameters to tweak
max_iters = 20_001
eval_iters = 10
eval_interval = 5_000
n_embed = 512
block_size = 128
batch_size = 16 # Increased for better GPU utilization
learning_rate = 3e-4
n_head = 8
n_layer = 20
dropout = 0.2

num_generalist_experts = 2 # Always-on experts applied to every token
num_specialist_experts = 8 # Routed experts, top_k of these are picked per token
top_k = 2 # Number of specialist experts to route each token to

# Multi-Head Latent Attention (MLA) hyperparameters
rope_dim = 16 # Decoupled RoPE dimension per head
q_lora_rank = n_embed // 4 # Compressed latent dimension for queries
kv_lora_rank = n_embed // 4 # Compressed latent dimension for keys/values


vocab_size = 1000
num_merges = vocab_size - 256
class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        text_chunks = self.compiled_pattern.findall(text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges):
            stats = Counter()
            for chunk_ids in ids:
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] += 1
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx}")

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        all_ids = []
        for chunk in self.compiled_pattern.findall(text):
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = Counter(zip(chunk_ids, chunk_ids[1:]))
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = self._merge(chunk_ids, pair, self.merges[pair])
            all_ids.extend(chunk_ids)
        return all_ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            part_bytes.append(self.vocab[idx])
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")

cache_file = f"wikitext_bpe_cache_v2_{vocab_size}.pkl"

# cache_file = f"wikitext_regex_bpe_cache_{dataset_range}_{vocab_size}.pkl"
tokenizer = BPETokenizer()

if os.path.exists(cache_file):
    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    data = cache_data['data']
    tokenizer.merges = cache_data['merges']
    tokenizer.vocab = cache_data['vocab']
else:
    # print(f"Downloading and processing wikitext dataset (range: {dataset_range})...")
    textraw = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    sample = textraw['train']#.select(range(min(dataset_range, len(textraw['train']))))
    text = " ".join(sample["text"])

    print("Training regex BPE tokenizer...")
    tokenizer.train(text, vocab_size, verbose=True)

    print("Encoding dataset...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump({'data': data, 'merges': tokenizer.merges, 'vocab': tokenizer.vocab}, f)

def decode(ids):
    return tokenizer.decode(ids)

def encode(text):
    return tokenizer.encode(text)

print("vocab size ", vocab_size)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)




def get_batch(split):
    #generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
    x,y = x.to(device), y.to(device)
    return x,y
# xb, yb = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#trying out this rope class, used for MLA's decoupled RoPE pathway
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """
        Args:
            dim: The dimension of the head (head_dim). Must be even.
            max_seq_len: Initial maximum sequence length to precompute cache for.
            base: The theta base for frequency calculation (default: 10000).
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Calculate inverse frequencies: theta_i = base^(-2(i-1)/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute the cos and sin caches
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        # Generate position indices [0, 1, ..., seq_len - 1]
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)

        # Outer product to get frequencies for all positions: shape (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)

        # Duplicate columns to match the full head_dim: shape (seq_len, dim)
        # This handles the [cos, cos] and [sin, sin] alignment for rotate_half
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache tensors with shape format [1, 1, seq_len, dim] for easy broadcasting
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len: int):
        # Dynamically scale cache if a longer sequence is provided at inference
        if seq_len > self.cos_cached.shape[2]:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits the hidden dimension in half and rotates the chunks."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies RoPE to query and key tensors.
    Expected shapes for q and k: [batch_size, num_heads, seq_len, head_dim]
    Expected shapes for cos and sin: [1, 1, seq_len, head_dim]
    """
    # Standard 2D rotation formula: x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

###end of rope code


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek-V2 style): queries and keys/values are
    each compressed down to a small latent vector and up-projected back out,
    which shrinks the KV cache at inference time to just the latent (instead of
    full per-head K/V). Position information is injected via a small decoupled
    RoPE pathway that bypasses the compression, since RoPE doesn't commute with
    the down/up projections.
    """
    def __init__(self, n_embed, num_heads, head_size, rope):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.rope = rope

        # Query pathway: down to latent, then up to content + decoupled rope
        self.q_down = nn.Linear(n_embed, q_lora_rank, bias=False)
        self.q_up = nn.Linear(q_lora_rank, num_heads * head_size, bias=False)
        self.q_rope = nn.Linear(q_lora_rank, num_heads * rope_dim, bias=False)

        # KV pathway: down to latent (this is what you'd cache at inference), then
        # up to key/value content. The rope key is shared across heads and taken
        # directly from x, not from the compressed latent.
        self.kv_down = nn.Linear(n_embed, kv_lora_rank, bias=False)
        self.kv_up = nn.Linear(kv_lora_rank, num_heads * head_size * 2, bias=False)
        self.k_rope = nn.Linear(n_embed, rope_dim, bias=False)

        self.out_proj = nn.Linear(num_heads * head_size, n_embed)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # --- QUERY PATHWAY ---
        c_q = self.q_down(x)
        q_c = self.q_up(c_q).view(B, T, self.num_heads, self.head_size)
        q_r = self.q_rope(c_q).view(B, T, self.num_heads, rope_dim)

        # --- KV PATHWAY ---
        c_kv = self.kv_down(x)
        kv_content = self.kv_up(c_kv).view(B, T, self.num_heads, self.head_size * 2)
        k_c, v_c = kv_content.split(self.head_size, dim=-1)

        # Independent RoPE key projection, shared across all heads
        k_r = self.k_rope(x).view(B, T, 1, rope_dim)

        # --- APPLY DECOUPLED RoPE ---
        q_r = q_r.transpose(1, 2) # (B, Heads, T, rope_dim)
        k_r = k_r.transpose(1, 2) # (B, 1, T, rope_dim)

        cos, sin = self.rope(q_r, seq_len=T)
        q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin)

        q_r = q_r.transpose(1, 2)
        k_r = k_r.transpose(1, 2)

        # Broadcast the shared k_r to all heads
        k_r = k_r.expand(-1, -1, self.num_heads, -1)

        # --- CONCATENATE CONTENT + RoPE ---
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2) # (B, Heads, T, head_size + rope_dim)
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2) # (B, Heads, T, head_size + rope_dim)
        v = v_c.transpose(1, 2)                           # (B, Heads, T, head_size)

        # --- ATTENTION ---
        scale = (self.head_size + rope_dim) ** -0.5
        wei = q @ k.transpose(-2, -1) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v # (B, Heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.out_proj(out))
        return out


class Expert(nn.Module):
    """ An expert network, which is a simple feed-forward network. """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class GeneralistSpecialistMoE(nn.Module):
    """
    A Mixture of Experts feed-forward layer split into two expert pools
    (DeepSeekMoE-style):

      - generalist experts: always applied to every token, summed unconditionally.
      - specialist experts: top_k routed per token via a gating network, exactly
        like a standard MoE gate.

    Args:
        n_embed (int): The embedding dimension.
        num_generalist_experts (int): Number of always-on experts.
        num_specialist_experts (int): Number of routed experts.
        top_k (int): The number of specialist experts to route each token to.
    """
    def __init__(self, n_embed, num_generalist_experts, num_specialist_experts, top_k):
        super().__init__()
        self.num_specialist_experts = num_specialist_experts
        self.top_k = top_k

        self.generalist_experts = nn.ModuleList([Expert(n_embed) for _ in range(num_generalist_experts)])
        self.specialist_experts = nn.ModuleList([Expert(n_embed) for _ in range(num_specialist_experts)])

        # The gating network only routes among the specialist experts
        self.gate = nn.Linear(n_embed, num_specialist_experts)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, n_embed) -> b, t, c
        b, t, c = x.shape

        # Flatten the input for token-wise processing
        x_flat = x.view(-1, c) # -> (b*t, c)

        # Generalist experts run unconditionally on every token and are summed.
        generalist_out = torch.zeros_like(x_flat)
        for expert in self.generalist_experts:
            generalist_out = generalist_out + expert(x_flat)

        # 1. Gating: Get logits for each token and each specialist expert
        gate_logits = self.gate(x_flat) # -> (b*t, num_specialist_experts)

        # 2. Routing: Find the top_k experts with the highest logits for each token
        # topk returns a tuple of (values, indices)
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1) # -> (b*t, top_k)

        # 3. Normalize the weights of the selected experts using softmax
        top_k_weights = F.softmax(top_k_logits, dim=-1) # -> (b*t, top_k)

        # 4. Combine results: Weighted sum of expert outputs
        specialist_out = torch.zeros_like(x_flat)

        # Get the indices of tokens and experts to be processed
        flat_token_indices = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.top_k)
        flat_expert_indices = top_k_indices.view(-1)

        # Group inputs by expert to process them in batches
        # This is more efficient than looping through each token
        for i in range(self.num_specialist_experts):
            # Find which tokens are routed to this expert
            token_mask = (flat_expert_indices == i)
            if token_mask.any():
                # Get the indices of the tokens for the current expert
                expert_token_indices = flat_token_indices[token_mask]

                # Get the input for this expert
                expert_input = x_flat[expert_token_indices]

                # Process the input with the expert
                expert_output = self.specialist_experts[i](expert_input)

                # Get the corresponding weights
                weights_for_expert = top_k_weights.view(-1)[token_mask]

                # Weight the expert's output
                weighted_output = expert_output * weights_for_expert.unsqueeze(1)

                # Add the weighted output back to the final result tensor
                # index_add_ is an efficient in-place scatter-add operation
                specialist_out.index_add_(0, expert_token_indices, weighted_output)

        # Reshape the output back to the original input shape
        final_output_flat = generalist_out + specialist_out
        return final_output_flat.view(b, t, c)

# --- Updated Block ---
# The Transformer Block now uses Multi-Head Latent Attention and the
# generalist/specialist MoE layer instead of plain attention + a single MoE pool.

class Block(nn.Module):
    """ Transformer block: communication (MLA) followed by computation (generalist/specialist MoE) """
    def __init__(self, n_embed, n_head, rope, num_generalist_experts, num_specialist_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadLatentAttention(n_embed, n_head, head_size, rope) # <-- CHANGED
        self.moe = GeneralistSpecialistMoE(n_embed, num_generalist_experts, num_specialist_experts, top_k) # <-- CHANGED
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # The architecture with residual connections remains the same
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x)) # <-- CHANGED
        return x

class Transformer(nn.Module):

    def __init__(self, num_generalist_experts, num_specialist_experts, top_k):
        super().__init__()
        #each token reads off the logits for the next tokenfrom a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # No absolute position embedding: positions come from RoPE inside MLA.
        self.rope = RotaryEmbedding(dim=rope_dim, max_seq_len=block_size)
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head=n_head, rope=self.rope,
                  num_generalist_experts=num_generalist_experts,
                  num_specialist_experts=num_specialist_experts, top_k=top_k)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b,t = idx.shape
        #idx and targets are both (b,t) tensor of integers
        x = self.token_embedding_table(idx) #(b,t,c)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the  last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = Transformer(num_generalist_experts, num_specialist_experts, top_k)
total_params = sum(p.numel() for p in model.parameters())
print('size of model',total_params)
m = model.to(device)

# logits, loss = m(xb,yb)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

start_time = time.time()

for iter in range(max_iters):

    #every once in awhile evaluate the loss on traon and val sets
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample batch of data
    xb, yb = get_batch('train')
    
    #evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

context = idx=torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
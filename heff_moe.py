"""
Full model + training script, same architecture as huge_eff_intr_dim_wiki.py
(intrinsic-dimension weight sharing + MLA + RoPE, trained on the same wikitext
data), but with the feed-forward Mixture-of-Experts layer split into two
weight-shared expert pools instead of one plain per-layer MoE:

  - generalist experts: always applied to every token (DeepSeekMoE-style shared experts).
  - specialist experts: top-k routed per token via a gating network.

Each expert "slot" (generalist or specialist) owns one SharedMaskedGroupLinear
up/down pair. That pair is created once and handed to every layer; each layer
rotates through a subset of the pair's weight matrices and trains a disjoint
masked subset of it, exactly like the plain FeedForward sharing scheme in
efficientintrdim_incl_ffwd.py.
"""
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

# intrinsic dimension adjustments
num_matrices = 2
percentweights = 0.2  # amount each layer updates of its assigned matrix

# separate intrinsic dimension settings for the expert banks (generalist + specialist)
expert_num_matrices = 4
expert_percentweights = 0.4  # amount each layer updates of its assigned expert matrix


vocab_size = 1_000
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

tokenizer = BPETokenizer()

def decode(ids):
    return tokenizer.decode(ids)

def encode(text):
    return tokenizer.encode(text)


def load_or_train_tokenizer_and_data():
    """Loads cached BPE tokenizer + encoded dataset, or trains/encodes from scratch."""
    cache_file = f"wikitext_bpe_cache_v2_{vocab_size}.pkl"

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        loaded_data = cache_data['data']
        tokenizer.merges = cache_data['merges']
        tokenizer.vocab = cache_data['vocab']
    else:
        textraw = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
        sample = textraw['train']
        text = " ".join(sample["text"])

        print("Training regex BPE tokenizer...")
        tokenizer.train(text, vocab_size, verbose=True)

        print("Encoding dataset...")
        loaded_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        print(f"Saving cache to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump({'data': loaded_data, 'merges': tokenizer.merges, 'vocab': tokenizer.vocab}, f)

    return loaded_data


def get_batch(split):
    #generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

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


#trying out this rope class, experimental so far
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


class SharedMaskedGroupLinear(nn.Module):
    """
    Implements the C++ intrinsic dimensionality logic in PyTorch.
    Uses multiple matrices. Layers rotate through the matrices,
    each using a predefined disjoint random subset of the selected matrix's weights.
    """
    def __init__(self, in_features, out_features, num_layers, num_matrices, percentweights, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_matrices = num_matrices

        # We hold num_matrices shared weight matrices
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, in_features))
            for _ in range(num_matrices)
        ])
        for w in self.weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        if bias:
            self.biases = nn.Parameter(torch.Tensor(num_layers, out_features))  # independent bias per layer
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.biases, -bound, bound)
        else:
            self.register_parameter('biases', None)

        num_elements = in_features * out_features
        num_subset = int(num_elements * percentweights)

        self.layer_to_matrix = []

        # Partition permutations disjointly for the number of layers that share each matrix
        for m in range(num_matrices):
            layers_for_m = [i for i in range(num_layers) if i % num_matrices == m]

            # Start with a random shuffle of all flat indices
            indices = torch.randperm(num_elements)
            for j, layer_idx in enumerate(layers_for_m):
                mask = torch.zeros(num_elements, dtype=torch.bool)
                start = j * num_subset
                end = (j + 1) * num_subset
                end = min(end, num_elements)

                if start < num_elements:
                    layer_indices = indices[start:end]
                    mask[layer_indices] = True

                mask = mask.view(out_features, in_features)
                self.register_buffer(f'mask_{layer_idx}', mask)

        # Register mapping matrix assignment for each layer
        for i in range(num_layers):
            self.layer_to_matrix.append(i % num_matrices)

    def forward(self, x, layer_idx):
        matrix_idx = self.layer_to_matrix[layer_idx]
        w = self.weights[matrix_idx]

        # Obtain the assigned mask and implicitly cast to parameter dtype
        mask = getattr(self, f'mask_{layer_idx}').to(w.dtype)

        # Forward pass uses the literal values of W due to the structure (w*1 + w*0 = w) but
        # detaching the complement stops gradients from accumulating in the ignored space!
        w_fake = w * mask + w.detach() * (1.0 - mask)

        b = None if self.biases is None else self.biases[layer_idx]
        return F.linear(x, w_fake, b)


def count_unshared_equivalent_params(model):
    """
    Total parameter count if every layer owned its own independent weight matrix
    instead of sharing/masking a pool of `num_matrices` matrices per SharedMaskedGroupLinear.
    Recomputed from each shared linear's current in/out/num_layers/num_matrices, so it
    tracks hyperparameter changes automatically.
    """
    total = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, SharedMaskedGroupLinear):
            per_matrix = module.in_features * module.out_features
            actual = module.num_matrices * per_matrix
            independent = module.num_layers * per_matrix
            total += independent - actual
    return total


class SharedExpertBank(nn.Module):
    """
    Holds one SharedMaskedGroupLinear up/down pair per expert slot. The bank is
    built once by the model and handed to every layer, so expert i's weights are
    shared (and masked per-layer) across the whole network instead of each layer
    getting its own independent expert.
    """
    def __init__(self, n_embed, num_experts, n_layer, num_matrices, percentweights, hidden_mult=2):
        super().__init__()
        hidden = hidden_mult * n_embed
        self.up = nn.ModuleList([
            SharedMaskedGroupLinear(n_embed, hidden, n_layer, num_matrices, percentweights, bias=False)
            for _ in range(num_experts)
        ])
        self.down = nn.ModuleList([
            SharedMaskedGroupLinear(hidden, n_embed, n_layer, num_matrices, percentweights, bias=False)
            for _ in range(num_experts)
        ])

    def __len__(self):
        return len(self.up)

    def projections(self, expert_idx):
        return self.up[expert_idx], self.down[expert_idx]


class SharedExpert(nn.Module):
    """A single expert whose up/down projections are shared, layer-masked weights."""
    def __init__(self, layer_idx, shared_up, shared_down, dropout):
        super().__init__()
        self.layer_idx = layer_idx
        self.shared_up = shared_up
        self.shared_down = shared_down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.shared_up(x, self.layer_idx)
        x = F.relu(x)
        x = self.shared_down(x, self.layer_idx)
        x = self.dropout(x)
        return x


class SharedMixtureOfExperts(nn.Module):
    """
    A Mixture of Experts feed-forward layer built from weight-shared experts.

    Args:
        n_embed (int): The embedding dimension.
        layer_idx (int): Index of the layer this instance belongs to.
        generalist_bank (SharedExpertBank): experts run on every token, unrouted.
        specialist_bank (SharedExpertBank): experts routed top-k per token.
        top_k (int): Number of specialist experts to route each token to.
        dropout (float): Dropout applied inside each expert.
    """
    def __init__(self, n_embed, layer_idx, generalist_bank, specialist_bank, top_k, dropout):
        super().__init__()
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.num_specialists = len(specialist_bank)

        self.generalist_experts = nn.ModuleList([
            SharedExpert(layer_idx, *generalist_bank.projections(i), dropout)
            for i in range(len(generalist_bank))
        ])
        self.specialist_experts = nn.ModuleList([
            SharedExpert(layer_idx, *specialist_bank.projections(i), dropout)
            for i in range(self.num_specialists)
        ])

        # Gating network: picks which specialist experts handle each token.
        self.gate = nn.Linear(n_embed, self.num_specialists)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, n_embed) -> b, t, c
        b, t, c = x.shape
        x_flat = x.view(-1, c)  # -> (b*t, c)

        # Generalist experts run unconditionally on every token and are summed.
        generalist_out = torch.zeros_like(x_flat)
        for expert in self.generalist_experts:
            generalist_out = generalist_out + expert(x_flat)

        # Specialist experts: top-k routing, same scheme as a standard MoE gate.
        gate_logits = self.gate(x_flat)  # -> (b*t, num_specialists)
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)  # -> (b*t, top_k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # -> (b*t, top_k)

        specialist_out = torch.zeros_like(x_flat)
        flat_token_indices = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.top_k)
        flat_expert_indices = top_k_indices.view(-1)

        for i in range(self.num_specialists):
            token_mask = (flat_expert_indices == i)
            if token_mask.any():
                expert_token_indices = flat_token_indices[token_mask]
                expert_input = x_flat[expert_token_indices]
                expert_output = self.specialist_experts[i](expert_input)

                weights_for_expert = top_k_weights.view(-1)[token_mask]
                weighted_output = expert_output * weights_for_expert.unsqueeze(1)

                specialist_out.index_add_(0, expert_token_indices, weighted_output)

        final_output_flat = generalist_out + specialist_out
        return final_output_flat.view(b, t, c)


class MultiHeadLatentAttentionBatch(nn.Module):
    """Multi-Head Latent Attention using SharedMaskedGroupLinear projections."""
    def __init__(self, num_heads, head_size, layer_idx,
                 shared_q_down, shared_q_up, shared_q_rope,
                 shared_kv_down, shared_kv_up, shared_k_rope,
                 shared_out_proj, rope, rope_dim):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_size = head_size
        self.rope_dim = rope_dim
        self.rope = rope

        # Unpack the shared linear projections
        self.shared_q_down = shared_q_down
        self.shared_q_up = shared_q_up
        self.shared_q_rope = shared_q_rope
        self.shared_kv_down = shared_kv_down
        self.shared_kv_up = shared_kv_up
        self.shared_k_rope = shared_k_rope
        self.shared_out_proj = shared_out_proj

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # --- QUERY PATHWAY ---
        # 1. Compress to latent query space
        c_q = self.shared_q_down(x, self.layer_idx)
        # 2. Up-project to Content and RoPE
        q_c = self.shared_q_up(c_q, self.layer_idx).view(B, T, self.num_heads, self.head_size)
        q_r = self.shared_q_rope(c_q, self.layer_idx).view(B, T, self.num_heads, self.rope_dim)

        # --- KV PATHWAY ---
        # 1. Compress to latent KV bottleneck (This is what you'd cache during inference!)
        c_kv = self.shared_kv_down(x, self.layer_idx)
        # 2. Up-project to get Key and Value content
        kv_content = self.shared_kv_up(c_kv, self.layer_idx).view(B, T, self.num_heads, self.head_size * 2)
        k_c, v_c = kv_content.split(self.head_size, dim=-1)

        # 3. Independent RoPE Key projection (Shared across all heads logically)
        k_r = self.shared_k_rope(x, self.layer_idx).view(B, T, 1, self.rope_dim)

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
        # Scale by combined dimensions
        scale = (self.head_size + self.rope_dim) ** -0.5
        wei = q @ k.transpose(-2, -1) * scale

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Values map purely to the content dimension
        out = wei @ v # (B, Heads, T, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.shared_out_proj(out, self.layer_idx)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_idx,
                 shared_q_down, shared_q_up, shared_q_rope,
                 shared_kv_down, shared_kv_up, shared_k_rope,
                 shared_out_proj, rope, rope_dim,
                 generalist_bank, specialist_bank):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadLatentAttentionBatch(
            n_head, head_size, layer_idx,
            shared_q_down, shared_q_up, shared_q_rope,
            shared_kv_down, shared_kv_up, shared_k_rope,
            shared_out_proj, rope, rope_dim
        )
        self.ffwd = SharedMixtureOfExperts(n_embed, layer_idx, generalist_bank, specialist_bank, top_k, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # MLA Hyperparameters
        head_size = n_embed // n_head
        self.rope_dim = 32 # Decoupled RoPE dimension
        self.q_lora_rank = 128
        self.kv_lora_rank = 128

        # 1. RoPE is now sized strictly for rope_dim
        self.rope = RotaryEmbedding(dim=self.rope_dim, max_seq_len=block_size)

        # 2. Instantiate the 7 SharedMaskedGroupLinear pathways
        # Query pathways
        self.shared_q_down = SharedMaskedGroupLinear(n_embed, self.q_lora_rank, n_layer, num_matrices, percentweights, bias=False)
        self.shared_q_up = SharedMaskedGroupLinear(self.q_lora_rank, n_head * head_size, n_layer, num_matrices, percentweights, bias=False)
        self.shared_q_rope = SharedMaskedGroupLinear(self.q_lora_rank, n_head * self.rope_dim, n_layer, num_matrices, percentweights, bias=False)

        # KV pathways
        self.shared_kv_down = SharedMaskedGroupLinear(n_embed, self.kv_lora_rank, n_layer, num_matrices, percentweights, bias=False)
        self.shared_kv_up = SharedMaskedGroupLinear(self.kv_lora_rank, n_head * head_size * 2, n_layer, num_matrices, percentweights, bias=False)
        self.shared_k_rope = SharedMaskedGroupLinear(n_embed, self.rope_dim, n_layer, num_matrices, percentweights, bias=False)

        # Output projection
        self.shared_out_proj = SharedMaskedGroupLinear(n_head * head_size, n_embed, n_layer, num_matrices, percentweights, bias=False)

        # Weight-shared expert banks: generalist experts run on every token,
        # specialist experts are top_k routed. Both share weights across layers.
        # Uses its own num_matrices/percentweights, independent of the attention pathways.
        self.generalist_bank = SharedExpertBank(n_embed, num_generalist_experts, n_layer, expert_num_matrices, expert_percentweights)
        self.specialist_bank = SharedExpertBank(n_embed, num_specialist_experts, n_layer, expert_num_matrices, expert_percentweights)

        self.blocks = nn.Sequential(*[
            Block(
                n_embed, n_head, i,
                self.shared_q_down, self.shared_q_up, self.shared_q_rope,
                self.shared_kv_down, self.shared_kv_up, self.shared_k_rope,
                self.shared_out_proj, self.rope, self.rope_dim,
                self.generalist_bank, self.specialist_bank
            ) for i in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape

        # No absolute position embedding: positions come from RoPE inside attention.
        x = self.token_embedding_table(idx)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # NOTE: Standard autoregressive generation.
        # Caching logic is not implemented here.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    data = load_or_train_tokenizer_and_data()

    print("vocab size ", vocab_size)
    n = int(0.9*len(data))
    train_data = data[:n]
    test_data = data[n:]

    torch.manual_seed(1337)

    model = Transformer()
    total_params = sum(p.numel() for p in model.parameters())
    print('size of model', total_params)
    print('size of model if weights were not reused across layers', count_unshared_equivalent_params(model))
    m = model.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    start_time = time.time()

    for iter in range(max_iters):

        #every once in awhile evaluate the loss on train and val sets
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

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"Dataset not found at . Please adjust path or ensure dataset exists.")
    text = "Placeholder text to prevent crash if file not found. " * 100

print('device is: ', device)

# parameters to tweak
max_iters = 5_001
eval_iters = 100
eval_interval = 1_000
n_embed = 256
block_size = 32
batch_size = 12
learning_rate = 5e-4
n_head = 4
n_layer = 6  # scaled to 6 layers
dropout = 0.2

# intrinsic dimension adjustments
num_matrices = 2
percentweights = 0.33  # amount each layer updates of its assigned matrix

vocab_size = 400
num_merges = vocab_size - 256

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i<len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+= 2
        else:
            newids.append(ids[i])
            i+=1
    return newids

ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:  # safety
        break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("merged")
print('len: ',len(ids))

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0,p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
    
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors='replace')
    return text

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

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
            self.biases = nn.Parameter(torch.Tensor(num_layers, out_features)) # independent bias per layer
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


class MultiheadAttentionBatch(nn.Module):
    """Refactored multihead attention to use the shared projections."""
    def __init__(self, num_heads, head_size, layer_idx, shared_query, shared_key, shared_value, shared_output_proj):
        super().__init__()
        self.layer_idx = layer_idx
        self.shared_query = shared_query
        self.shared_key = shared_key
        self.shared_value = shared_value
        self.shared_output_proj = shared_output_proj
        self.num_heads = num_heads
        self.head_size = head_size
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.shared_query(x, self.layer_idx).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Obtain Key and Value outputs utilizing the unified parameters
        k = self.shared_key(x, self.layer_idx).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.shared_value(x, self.layer_idx).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v 
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.shared_output_proj(out, self.layer_idx)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, layer_idx, shared_up, shared_down):
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

class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_idx, shared_query, shared_key, shared_value, shared_output_proj, shared_ff_up, shared_ff_down):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttentionBatch(n_head, head_size, layer_idx, shared_query, shared_key, shared_value, shared_output_proj)
        self.ffwd = FeedForward(n_embed, layer_idx, shared_ff_up, shared_ff_down)
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
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Intrinsic dimension elements
        self.shared_query = SharedMaskedGroupLinear(n_embed, n_embed, n_layer, num_matrices, percentweights, bias=False)
        self.shared_key = SharedMaskedGroupLinear(n_embed, n_embed, n_layer, num_matrices, percentweights, bias=False)
        self.shared_value = SharedMaskedGroupLinear(n_embed, n_embed, n_layer, num_matrices, percentweights, bias=False)
        self.shared_output_proj = SharedMaskedGroupLinear(n_embed, n_embed, n_layer, num_matrices, percentweights, bias=False)
        self.shared_ff_up = SharedMaskedGroupLinear(n_embed, 4*n_embed, n_layer, num_matrices, percentweights, bias=False)
        self.shared_ff_down = SharedMaskedGroupLinear(4*n_embed, n_embed, n_layer, num_matrices, percentweights, bias=False)
        
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head=n_head, layer_idx=i, shared_query=self.shared_query, shared_key=self.shared_key, shared_value=self.shared_value, shared_output_proj=self.shared_output_proj, shared_ff_up=self.shared_ff_up, shared_ff_down=self.shared_ff_down) 
            for i in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed) 
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b,t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

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

model = Transformer()
total_params = sum(p.numel() for p in model.parameters())
print('size of model (intrinsic dimensions adapted):', total_params)

m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

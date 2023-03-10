import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import datetime
from postprocessing import notes_to_midi
import json

# Hyperparameters
batch_size = 64  # no. independent sequences processed in parallel
block_size = 1024  # maximum context length for predictions
max_iters = 200
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 6
n_embd = 64 * n_head  # -> each head is 64-dimensional
n_layer = 6
dropout = 0.2
# ------------
n_gen_tokens = 1000
n_midi_files = 50
model_path = 'models/interleave_2023_02_25_02_04_29'


# Load notes into numpy array
data = np.fromfile('data/notes_8.bin', dtype=np.dtype('uint8'))
# Every 8 values contain 7 zeros and 1 value
# -> reshape and remove zeros
data = data.reshape((data.shape[0] // 8, 8))
data = data[:, -1].reshape(data.shape[0] // 2, 2)


# Interleave interval and duration values
data = data.reshape((data.shape[0] * 2,))

print(f'Data loaded: {data.shape}')

unique_tokens = list(np.unique(data).tolist())
vocab_size = len(unique_tokens)

ntoi = {unique_tokens[i]: i for i in range(len(unique_tokens))}
iton = {i: unique_tokens[i] for i in range(len(unique_tokens))}
encode = lambda n: ntoi[n]
decode = lambda i: iton[i]

data = np.array(list(map(encode, data)))

# Uncomment for faster computation when running locally (comment line above)
# with open('data/data_encoded.txt', 'r') as f:
#     data = f.read()
#     data = json.loads(data)
#     data = np.array(data)

print(f'Vocab size: {vocab_size}')


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # not a model parameter, therefore a PyTorch 'buffer'
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ('affinities') (normalise by root -> scaled attention)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # ModuleList allows for multiple heads in parallel (inside a list)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projection layer going back into residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate all outputs in list over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity (ReLU)"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # network gets wider -> additional computation
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer going back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # communication
        self.ffwd = FeedForward(n_embd)  # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # skip connections + pre-norm formulation ("modern" deviation from original paper)
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # each position also gets an embedding vector
        # sequential transformer blocks: intersperse communication -> computation over and over again
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # map back to characters
        self.lm_head = nn.Linear(n_embd, vocab_size)  # maps from embedding size to vocab size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C), contains not only identities, but also positional information
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last {block_size} tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits/temperature, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = Transformer()
m = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()

print(f'Loaded model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')


# Generate MIDI files
for _ in range(n_midi_files):
    context = torch.Tensor([[encode(128), encode(0)]])
    context = context.to(device)
    context = context.to(torch.long)
    seq = m.generate(context, max_new_tokens=n_gen_tokens, temperature=1.1)[0].tolist()
    seq = list(map(decode, seq))

    # Remove offset from interval values (by subtracting 128)
    seq = np.array(seq).reshape((len(seq) // 2), 2)
    seq = seq - np.array([128, 0])

    dt_now = datetime.datetime.now()
    out_path = f'interleave/interleave_{str(dt_now.date()).replace("-", "_")}_{str(dt_now.time()).split(".")[0].replace(":", "_")}'
    notes_to_midi(seq, path=out_path)

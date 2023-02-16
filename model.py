import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4  # decrease lr for self-attention
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 768  # 384 / 6 = 64 -> each head is 64-dimensional (this is standard)
n_head = 6
n_layer = 6  # 6 transformer blocks
dropout = 0.2
# ------------

# Load notes into numpy array
data = np.fromfile('notes_8.bin', dtype=np.dtype('uint8'))
data = data.reshape((data.shape[0] // 8, 8))
# Every 8 values contain 7 zeros and 1 value
# -> reshape and remove zeros
data = data[:, -1].reshape(data.shape[0] // 2, 2)
print(f'Data loaded: {data.shape}')
# First 90% of notes are train, rest are val
train_data, val_data = np.array_split(data, [int(data.shape[0] * 0.9)])

vocab_size = 256 + 32  # 256 intervals + 32 durations


def get_batch(phase='train'):
    data = train_data if phase == 'train' else val_data
    # Randomly generated offsets into training set, each at a multiple of 13
    ix = np.random.randint(0, data.shape[0] - block_size, (batch_size,))
    x = np.stack([data[x:x+block_size] for x in ix])
    y = np.stack([data[x+1:x+block_size+1] for x in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    # TODO: this line is probably wrong
    x, y = x.to(torch.int64), y.to(torch.int64)
    return x, y


x, y = get_batch()
x_ivl, x_dur = torch.split(x, 1, dim=2)
x_ivl = x_ivl[:, :, 0]
x_dur = x_dur[:, :, 0]


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # TODO: Intervals/durations?
            X, Y = get_batch(split)
            x_ivl, x_dur = torch.split(X, 1, dim=2)
            x_ivl = x_ivl[:, :, 0]
            x_dur = x_dur[:, :, 0]
            y_ivl, y_dur = torch.split(Y, 1, dim=2)
            y_ivl = y_ivl[:, :, 0]
            y_dur = y_dur[:, :, 0]

            logits, loss = model(x_ivl, y_ivl)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # communication
        self.ffwd = FeedForward(n_embd)  # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # skip connections + pre-norm formulation (modern deviation from original paper)
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
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

        # idx and targets are both (B,T) tensor of integers
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

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last {block_size} tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = Transformer()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # TODO: How to train for both interval and duration?
    # TODO: try training on entire seq tensors (interval + duration simultaneously)

    # sample a batch of data
    xb, yb = get_batch('train')
    x_ivl, x_dur = torch.split(xb, 1, dim=2)
    x_ivl = x_ivl[:, :, 0]
    x_dur = x_dur[:, :, 0]
    y_ivl, y_dur = torch.split(yb, 1, dim=2)
    y_ivl = y_ivl[:, :, 0]
    y_dur = y_dur[:, :, 0]

    # evaluate the loss
    logits, loss = model(x_ivl, y_ivl)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(m.generate(context, max_new_tokens=500)[0].tolist())

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import mido
# import libraries.mido as mido
import datetime

# Hyperparameters
batch_size = 64  # no. independent sequences processed in parallel
block_size = 1024  # maximum context length for predictions
max_iters = 25000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 6
n_embd = 64 * n_head  # -> each head is 64-dimensional
n_layer = 6
dropout = 0.2
# ------------
n_gen_tokens = 10000

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

print(f'Vocab size: {vocab_size}')

# First 90% of notes are train, rest are val
train_data, val_data = np.array_split(data, [int(data.shape[0] * 0.9)])

# For model log file
t0 = datetime.datetime.now()
loss_history = ""


def get_batch(phase='train'):
    data = train_data if phase == 'train' else val_data
    # Randomly generated offsets into training set, each at a multiple of 13
    ix = np.random.randint(0, (data.shape[0] - block_size) / 2, (batch_size,)) * 2
    # Uncomment to overfit
    # ix = np.zeros((batch_size,), dtype=np.int8)
    x = np.stack([data[x:x+block_size] for x in ix])
    y = np.stack([data[x+1:x+block_size+1] for x in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x, y = x.to(torch.int64), y.to(torch.int64)
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Evaluate loss every [eval_interval] iterations
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        loss_history += f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n"

    # sample a batch of data
    x, y = get_batch('train')

    # evaluate the loss
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# For timestamping + training time calculation
dt_now = datetime.datetime.now()

# Save model to disk
model_path = f'interleave_{str(dt_now.date()).replace("-", "_")}_{str(dt_now.time()).split(".")[0].replace(":", "_")}'
torch.save(model.state_dict(), model_path)


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


# Write model info to file
with open(f'{model_path}_hyperparameters.txt', 'w') as f:
    f.write(model_path + '\n\n')
    f.write('------------ Model hyperparameters ------------\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Block size: {block_size}\n')
    f.write(f'Embedding dimensions: {n_embd}\n')
    f.write(f'No. heads: {n_head}\n')
    f.write(f'No. layers: {n_layer}\n')
    f.write(f'Learning rate: {learning_rate}\n')
    f.write(f'Dropout: {dropout}\n\n')

    f.write('-------------- Training details --------------\n')
    f.write(f'Training time: {strfdelta(dt_now - t0, "{days} days {hours} hours {minutes} minutes {seconds} seconds")}\n')
    f.write(f'Max iters: {max_iters}\n')
    f.write(f'Eval interval: {eval_interval}\n')
    f.write(f'Eval iters: {eval_iters}\n')
    f.write(f'Device: {device}\n\n')

    f.write('-------------------- Loss --------------------\n')
    f.write(loss_history)

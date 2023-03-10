import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import datetime
import math

# Hyperparameters
batch_size = 4  # no. independent sequences processed in parallel
block_size = 16  # maximum context length for predictions
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 1
n_embd = 64 * n_head  # -> each head is 64-dimensional
n_layer = 1
dropout = 0.05
# ------------

# Load notes into numpy array
data = np.fromfile('data/notes_8.bin', dtype=np.dtype('uint8'))
# Every 8 values contain 7 zeros and 1 value
# -> reshape and remove zeros
data = data.reshape((data.shape[0] // 8, 8))
data = data[:, -1].reshape(data.shape[0] // 2, 2)

print(f'Data loaded: {data.shape}')


# Pitch value range in dataset is [46, 210] -> subtract 46 -> range is [0, 164]
def encode(d):
    if isinstance(d, np.ndarray) or isinstance(d, torch.Tensor):
        d[:, 0] -= 46
    elif isinstance(d, int):
        d -= 46
    return d


def decode(d):
    if isinstance(d, np.ndarray) or isinstance(d, torch.Tensor):
        d[:, 0] += 46
    elif isinstance(d, int):
        d += 46
    return d


data = encode(data)

unique_tokens = np.unique(data)
vocab_size = unique_tokens.shape[0]

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
        losses = torch.zeros((eval_iters, 2))
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, _, loss = model(X, Y)
            # losses[k] = loss.item()
            losses[k] = loss
        out[split] = torch.mean(losses, 0, True)
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
        self.pitch_embedding_table = nn.Embedding(vocab_size, n_embd // 2)
        self.dur_embedding_table = nn.Embedding(33, n_embd // 2)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # each position also gets an embedding vector
        # sequential transformer blocks: intersperse communication -> computation over and over again
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # map back to characters
        self.lm_head_pitch = nn.Linear(n_embd, vocab_size)   # maps from embedding size to vocab size
        self.lm_head_dur = nn.Linear(vocab_size + n_embd, vocab_size)  # maps from embedding size to vocab size

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # Split idx into pitch and dur
        idx_p, idx_d = idx[:, :, 0], idx[:, :, 1]
        B, T, _ = idx.shape  # ignore 2 in (B, T, 2)

        # idx and targets are both (B,T) tensors of integers
        pitch_emb = self.pitch_embedding_table(idx_p)
        dur_emb = self.dur_embedding_table(idx_d)
        model_input = torch.cat((pitch_emb, dur_emb), dim=-1)

        # Add positional information
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        model_input = model_input + pos_emb

        # Input data to blocks
        activations = self.blocks(model_input)  # (B, T, C)
        activations = self.ln_f(activations)  # (B, T, C)

        logits_pitch = self.lm_head_pitch(activations)  # (B, T, vocab_size)

        logits_pitch = logits_pitch.permute(*torch.arange(logits_pitch.ndim - 1, -1, -1))
        activations = activations.permute(*torch.arange(activations.ndim - 1, -1, -1))

        # Concatenate into dur_input and transpose back
        dur_input = torch.cat((logits_pitch, activations))
        dur_input = dur_input.permute(*torch.arange(dur_input.ndim - 1, -1, -1))
        # Transpose pitch logits back
        logits_pitch = logits_pitch.permute(*torch.arange(logits_pitch.ndim - 1, -1, -1))

        logits_dur = self.lm_head_dur(dur_input)  # (B, T, vocab_size)

        if targets is None:
            loss = None
            loss_pair = None
        else:
            # Split targets into pitch and dur
            targets_pitch, targets_dur = targets[:, :, 0], targets[:, :, 1]

            # Pitch loss
            B, T, C = logits_pitch.shape
            logits_pitch = logits_pitch.view(B * T, C)
            targets_pitch = targets_pitch.view(B * T)
            loss_pitch = F.cross_entropy(logits_pitch, targets_pitch)

            # Duration loss
            B, T, C = logits_dur.shape
            logits_dur = logits_dur.view(B * T, C)
            targets_dur = targets_dur.view(B * T)
            loss_dur = F.cross_entropy(logits_dur, targets_dur)

            # Loss is sum of both losses
            loss = loss_pitch + loss_dur
            loss_pair = torch.Tensor([loss_pitch, loss_dur])

        return (logits_pitch, logits_dur), loss, loss_pair

    def generate(self, idx, max_new_tokens, temperature):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last {block_size} tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            (logits_pitch, logits_dur), _, _ = self(idx_cond)
            # focus only on the last time step
            logits_pitch = logits_pitch[:, -1, :]  # becomes (B, C)
            logits_dur = logits_dur[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs_pitch = F.softmax(logits_pitch/temperature, dim=-1)  # (B, C)
            probs_dur = F.softmax(logits_dur/temperature, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next_pitch = torch.multinomial(probs_pitch, num_samples=1)  # (B, 1)
            idx_next_dur = torch.multinomial(probs_dur, num_samples=1)  # (B, 1)
            # clip durations to range [0, 32]
            idx_next_dur = min(idx_next_dur, torch.Tensor([[32]]))
            # append sampled index to the running sequence
            idx_next = torch.Tensor([[[idx_next_pitch, idx_next_dur]]])
            idx_next = idx_next.to(device)
            idx_next = idx_next.to(torch.long)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    model = Transformer()
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # Evaluate loss every [eval_interval] iterations
        if iter % eval_interval == 0:
            losses = estimate_loss()
            train_loss, val_loss = losses['train'], losses['val']
            loss_str = f"step {iter}: train loss {train_loss.sum():.4f} {*[int(x * 10000) / 10000 for x in train_loss.tolist()[0]],}, val loss {val_loss.sum():.4f} {*[int(x * 10000) / 10000 for x in val_loss.tolist()[0]],}\n"
            print(loss_str)
            loss_history += loss_str

        # sample a batch of data
        x, y = get_batch('train')

        # evaluate the loss
        logits, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # For timestamping + training time calculation
    dt_now = datetime.datetime.now()

    # Save model to disk
    model_path = f'2heads_{str(dt_now.date()).replace("-", "_")}_{str(dt_now.time()).split(".")[0].replace(":", "_")}'
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

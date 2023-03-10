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
n_gen_tokens = 1000
n_midi_files = 50
model_path = 'models/2heads_2023_03_01_08_06_50'


# Load notes into numpy array
data = np.fromfile('data/notes_8.bin', dtype=np.dtype('uint8'))
# Every 8 values contain 7 zeros and 1 value
# -> reshape and remove zeros
data = data.reshape((data.shape[0] // 8, 8))
data = data[:, -1].reshape(data.shape[0] // 2, 2)


# Tokenise
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

# Reshape back to (n, 2) array
data = data.reshape((data.shape[0] // 2, 2))

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
        self.pitch_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.dur_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # each position also gets an embedding vector
        # sequential transformer blocks: intersperse communication -> computation over and over again
        self.blocks_pitch = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f_pitch = nn.LayerNorm(n_embd)  # final layer norm
        # map back to characters
        self.lm_head_pitch = nn.Linear(n_embd, vocab_size)   # maps from embedding size to vocab siz

        # Duration transformer blocks
        self.blocks_dur = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f_dur = nn.LayerNorm(n_embd)  # final layer norm
        # self.lm_head_dur = nn.Linear(n_embd * 2, 32)  # maps from embedding size to vocab size
        self.lm_head_dur = nn.Linear(n_embd * 2, vocab_size)  # maps from embedding size to vocab size

        # Map pitch logits to embedding dims
        self.logits_to_embspace = nn.Linear(vocab_size, n_embd)

    def forward(self, idx, targets=None):
        idx_p, idx_d = idx[:, :, 0], idx[:, :, 1]
        B, T, _ = idx.shape  # ignore 2 in (B, T, 2)

        # idx and targets are both (B,T) tensors of integers
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        pitch_emb = self.pitch_embedding_table(idx_p)
        dur_emb = self.dur_embedding_table(idx_d)
        # Add positional information
        pitch_emb = pitch_emb + pos_emb
        dur_emb = dur_emb + pos_emb
        # tok_emb = torch.cat((self.pitch_embedding_table(idx_p), self.dur_embedding_table(idx_d)), dim=2)  # (B,T,C)
        # x = x.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        # print(x.shape)
        # x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * 2))
        # print(x.shape)

        # Input pitch data to blocks
        x = self.blocks_pitch(pitch_emb)  # (B, T, C)
        x = self.ln_f_pitch(x)  # (B, T, C)
        logits_pitch = self.lm_head_pitch(x)  # (B, T, vocab_size)

        # Input pitch logits + duration data to blocks
        logits_emb = self.logits_to_embspace(logits_pitch)
        x = torch.stack([logits_emb, dur_emb], dim=-1)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * 2))
        # TODO: does this need to run through entire transformer block with pitch data? or through its own transformer?
        # x = self.blocks_dur(x)  # (B, T, C)
        # x = self.ln_f_dur(x)  # (B, T, C)
        logits_dur = self.lm_head_dur(x)  # (B, T, vocab_size)

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
            logits_pitch = logits_pitch[:, -1, :] # becomes (B, C)
            logits_dur = logits_dur[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs_pitch = F.softmax(logits_pitch/temperature, dim=-1)  # (B, C)
            probs_dur = F.softmax(logits_dur/temperature, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next_pitch = torch.multinomial(probs_pitch, num_samples=1)  # (B, 1)
            idx_next_dur = torch.multinomial(probs_dur, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx_next = torch.Tensor([[[idx_next_pitch, idx_next_dur]]])
            idx_next = idx_next.to(device)
            idx_next = idx_next.to(torch.long)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = Transformer()
m = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()

print(f'Loaded model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')


# Generate MIDI files
for _ in range(n_midi_files):
    context = torch.Tensor([[[encode(128), encode(0)]]])
    context = context.to(device)
    context = context.to(torch.long)
    seq = m.generate(context, max_new_tokens=n_gen_tokens, temperature=1.1)[0]
    seq = torch.flatten(seq)
    seq = seq.tolist()
    seq = list(map(decode, seq))

    # Remove offset from interval values (by subtracting 128)
    seq = np.array(seq).reshape((len(seq) // 2), 2)
    seq = seq - np.array([128, 0])

    dt_now = datetime.datetime.now()
    out_path = f'2heads/2heads_{str(dt_now.date()).replace("-", "_")}_{str(dt_now.time()).split(".")[0].replace(":", "_")}'
    notes_to_midi(seq, path=out_path)

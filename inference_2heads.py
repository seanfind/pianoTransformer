import torch
import numpy as np
import datetime
from postprocessing import notes_to_midi
import model_2heads
from model_2heads import Transformer

# Hyperparameters (override values in model_2heads.py if necessary)
model_2heads.batch_size = 4  # no. independent sequences processed in parallel
model_2heads.block_size = 16  # maximum context length for predictions
model_2heads.n_head = 1
model_2heads.n_embd = 64 * model_2heads.n_head  # -> each head is 64-dimensional
model_2heads.n_layer = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_gen_tokens = 1000
n_midi_files = 1
model_path = '2heads_2023_03_10_16_57_47'


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


model = Transformer()
m = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.eval()

print(f'Loaded model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')


# Generate MIDI files
for _ in range(n_midi_files):
    context = torch.Tensor([[[encode(128), 0]]])
    context = context.to(device)
    context = context.to(torch.long)
    seq = m.generate(context, max_new_tokens=n_gen_tokens, temperature=1.05)[0]
    seq = decode(seq)

    # Remove offset from interval values (by subtracting 128)
    seq[:, 0] -= 128

    dt_now = datetime.datetime.now()
    out_path = f'2heads/2heads_{str(dt_now.date()).replace("-", "_")}_{str(dt_now.time()).split(".")[0].replace(":", "_")}'
    notes_to_midi(seq, path=out_path)

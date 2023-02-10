import torch
import numpy as np
import time

# Load notes into numpy array
data = np.fromfile('notes_8.bin', dtype=np.dtype('uint8'))
data = data.reshape((data.shape[0] // 8, 8))
data = data[:, -1].reshape(data.shape[0] // 2, 2)
print(f'Data loaded: {data.shape}')
train_data, val_data = np.array_split(data, [int(data.shape[0] * 0.9)])


batch_size = 64
block_size = 256


def get_batch(phase='train'):
    data = train_data if phase == 'train' else val_data
    # Randomly generated offsets into training set, each at a multiple of 13
    ix = np.random.randint(0, data.shape[0] - block_size, (batch_size,))
    x = np.stack([data[x:x+block_size] for x in ix])
    y = np.stack([data[x+1:x+block_size+1] for x in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    return x, y


# n = 1000
t0 = time.perf_counter()

for i in range(100000):
    x, y = get_batch()
    if (time.perf_counter() - t0) >= 1:
        print(f'Generated {i} examples in 1s')
        break

# print(f'Generated {n} examples in {(time.perf_counter() - t0)}s')

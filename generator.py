import array
import time
import mido
import os
import numpy as np
import random
import torch


def midi_to_notes(path: str):
    data = mido.MidiFile(path)
    seq = []

    # Read from MIDI file (2nd track contains notes)
    for msg in data.tracks[1]:
        n = msg.dict()
        try:
            seq.append([n['note'], n['velocity'], n['time']])
        except KeyError:
            seq.append([None, None, n['time']])

    # TODO: combine removal + interval system into single for-loop

    # Remove note_off and CC messages
    # for i in range(len(seq) - 1, -1, -1):
    #     n = seq[i]
    #     if n[0] is None or n[1] == 0:  # if is note_off or CC
    #         if i < len(seq) - 1:  # if i + 1 is within array
    #             if seq[i - 1] is not None:
    #                 seq[i - 1][2] += n[2]
    #         seq[i] = None

    for i in range(len(seq)):
        n = seq[i]
        if n[0] is None or n[1] == 0:  # if is note_off or CC
            if i < len(seq) - 1:  # if i + 1 is within array
                if seq[i + 1][0] is not None:
                    seq[i + 1][2] += n[2]
            seq[i] = None

    seq = [n for n in seq if n is not None]

    # Interval system
    for i in range(len(seq) - 1, -1, -1):
        if i >= 1:
            seq[i][0] -= seq[i - 1][0]
        else:
            seq[i][0] = 0

    # Remove velocity values
    seq = np.array(seq)
    seq = np.delete(seq, 1, 1)

    # Get most frequently occurring durations
    durs = seq.T[1].tolist()
    durs = sorted(durs, key=lambda x: durs.count(x), reverse=True)

    already_seen = []
    large_frequent_d = []

    for d in durs:
        if d > 80 and d not in already_seen:
            large_frequent_d.append(d)
            already_seen.append(d)
        if len(large_frequent_d) >= 5:
            break

    def reject_outliers(array, m=2.):
        data = np.array(array)
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        return data[s < m]

    # Remove outliers and calculate average of values occurring most frequently
    large_frequent_d = reject_outliers(large_frequent_d)
    fund = large_frequent_d.mean()

    # Normalise time values
    res = 2
    fund = fund // 2
    fund = np.array([1, 1 / (fund * res)])
    seq = np.multiply(seq, fund)
    seq = np.rint(seq)

    # Clip time durations at 32
    seq[seq[:, 1] > 32] = 32

    return seq


def notes_to_midi(seq, n0=60, dur=200, upscale=200, path='notes_out'):
    seq = seq.tolist()
    current_time = 0

    # Restore pitch and time values, make time values absolute
    for i in range(len(seq)):
        if i == 0:
            seq[i][0] += n0
        else:
            seq[i][0] += seq[i - 1][0]

        time_val = seq[i][1]
        seq[i][1] += current_time
        current_time += time_val
        seq[i][1] *= upscale

        seq[i].append(1)  # note_on indicator
        seq.append([seq[i][0], seq[i][1] + dur, 0])

    seq.sort(key=lambda x: x[1])

    for i in range(len(seq) - 2, -1, -1):
        if i >= 1:
            seq[i][1] -= seq[i - 1][1]
        else:
            seq[i][1] = 0

    midi_out = mido.MidiFile()
    out_track = mido.MidiTrack()
    midi_out.tracks.append(out_track)

    for n in seq:
        # clip notes to range [25, 80]
        note_number = int(n[0])
        if note_number > 80:
            note_number -= (((note_number - 80) // 12) + 1) * 12
        elif note_number < 25:
            note_number += (((25 - note_number) // 12) + 1) * 12
        out_track.append(mido.Message('note_on', note=note_number, velocity=int(100 * n[2]), time=int(n[1])))

    midi_out.save(f'./output/{path}.mid')


def notes_to_disk(seq):
    # intervals: -128 to +128 -> range of 256 -> 2^8
    # durations: 0 to 32 -> range of 32 -> 2^5

    seq = seq.astype('int8').tolist()
    print(seq[:15])
    # seq = [['{0:08b}'.format(ivl), '{0:05b}'.format(dur)] for ivl, dur in seq]
    # seq = [[str(ivl).encode(), str(dur).encode()] for ivl, dur in seq]
    # seq = [[binascii.hexlify(str(ivl).encode()), binascii.hexlify(str(dur).encode())] for ivl, dur in seq]
    # seq = [[to_hex(ivl), to_hex(dur)] for ivl, dur in seq]
    seq = [[ivl.to_bytes(8, 'big', signed=True), dur.to_bytes(5, 'big')] for ivl, dur in seq]
    # print(seq[:15])
    # seq = [[binascii.unhexlify(ivl), binascii.unhexlify(dur)] for ivl, dur in seq]
    # print(seq[:15])

    with open('bin_test', 'ab') as f:
        for b in seq:
            for v in b:
                f.write(v)


def notes_from_disk(path):
    with open(path, 'rb') as f:
        data = f.read()
        print('Bytes:', len(data))
        print('Blocks of 13:', len(data) // 13, '\nclean divide =', len(data) % 13 == 0)

        split_index = int(0.9 * len(data))  # first 90% of notes are train, rest is validation
        train_data = data[:split_index]
        val_data = data[split_index:]

        for i in range(len(data) // 13):
            note = data[(i * 13):(i * 13) + 13]
            ivl = int.from_bytes(note[:8], byteorder='big', signed=True)
            dur = int.from_bytes(note[8:], byteorder='big', signed=False)
            print(ivl, dur)
            if i > 20:
                break


# with open('notes', 'rb') as f:
#     data = f.read()
#     print('Bytes:', len(data))
#     print('Blocks of 13:', len(data) // 13, '\nclean divide =', len(data) % 13 == 0)
#
#     split_index = int(0.9 * len(data))  # first 90% of notes are train, rest is validation
#     train_data = data[:split_index]
#     val_data = data[split_index:]


batch_size = 32
block_size = 128 * 13  # n x 13-bit notes


# with batch = 32 and block = 128
# v1 at 72 examples/s
# v2 at 154 examples/s
def get_batch(phase='train'):
    data = train_data if phase == 'train' else val_data
    # Randomly generated offsets into training set, each at a multiple of 13
    ix = torch.multiply(torch.randint((len(data) // 13) - block_size, (batch_size,)), 13)
    note_batches = [data[i:i + block_size + 13] for i in ix]
    note_batches = [
        [
            [
                int.from_bytes(batch[(i * 13):(i * 13) + 13][:8], byteorder='big', signed=True),
                int.from_bytes(batch[(i * 13):(i * 13) + 13][8:], byteorder='big', signed=False)
            ]
            for i in range(0, len(batch) // 13)
        ] for batch in note_batches
    ]

    x = torch.Tensor(note_batches)[:, :-1]
    y = torch.Tensor(note_batches)[:, 1:]
    return x, y

# ------------ NumPy implementation ------------
# uses 8-bit dataset file with 128 added to all intervals
# this allows for all unsigned 8-bit integers


# Load notes into numpy array
data = np.fromfile('notes_8.bin', dtype=np.dtype('uint8'))
data = data.reshape((data.shape[0] // 8, 8))
data = data[:, -1].reshape(data.shape[0] // 2, 2)




# TODO: Currently only subset of dataset!!!!!!
data = data[:1000000]





print(f'Data loaded: {data.shape}')


# Convert [aaa, bb] into [aaabb]
def tokenise(x):
    return (x[0] * 100) + x[1]


data = np.array([tokenise(x) for x in data])
print(f'Data processed: {data.shape}')
print(data)
train_data, val_data = np.array_split(data, [int(data.shape[0] * 0.9)])


batch_size = 128
block_size = 256


# Much faster: ~2500 examples/s
def get_batch_8bit(phase='train'):
    data = train_data if phase == 'train' else val_data
    # Randomly generated offsets into training set, each at a multiple of 13
    ix = np.random.randint(0, data.shape[0] - block_size, (batch_size,))
    x = np.stack([data[x:x+block_size] for x in ix])
    y = np.stack([data[x+1:x+block_size+1] for x in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    return x, y


# TODO: make this faster (currently at ~60 iter/s)
# TODO: try implementing tokenisation in dataset loading?
def get_batch_bundle(phase='train'):
    data = train_data if phase == 'train' else val_data
    # Randomly generated offsets into training set, each at a multiple of 13
    ix = np.random.randint(0, data.shape[0] - block_size, (batch_size,))
    xy = np.stack([data[x:x+block_size+1] for x in ix])

    xy = xy.reshape((batch_size * (block_size + 1), 2))

    xy_list = xy.T.tolist()
    xy_tokens = []
    for i in range(batch_size*(block_size + 1)):
        ivl, dur = xy_list[0][i], xy_list[1][i]
        xy_tokens.append(int(f'{ivl}' + f'{dur:02d}'))
    xy_tokens = np.array(xy_tokens)
    xy_tokens = xy_tokens.reshape((batch_size, block_size + 1))
    x = xy_tokens[:, :-1]
    y = xy_tokens[:, 1:]
    return x, y


midi = os.listdir('midi')
# random.seed(10)
random.shuffle(midi)
if ".DS_Store" in midi:
    midi.remove(".DS_Store")

t0 = time.perf_counter()
n = 1000

# for m in midi[:n]:
#     print(m)
#     seq = midi_to_notes('./midi/' + m)
#     durs = seq.T[1]
#     # durs_s = sorted(durs, reverse=True)
#     # print(durs_s)
#     ivls = seq.T[0]
#     print(durs.max(), durs.min())
#     print(ivls.max(), ivls.min())
#     # notes_to_disk(seq)


# notes_from_disk('notes')


for i in range(10000):
    x, y = get_batch_8bit()
    # print('------------- x -------------')
    # print(x)
    # print('------------- y -------------')
    # print(y)
    if time.perf_counter() - t0 >= 1:
        print(f'n = {i}')
        break

print((time.perf_counter() - t0))

# notes_to_midi(seq)

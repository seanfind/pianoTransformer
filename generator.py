import time
import mido
import os
import numpy as np
import random


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
        for i in range(len(data) // 13):
            note = data[(i * 13):(i * 13) + 13]
            ivl = int.from_bytes(note[:8], byteorder='big', signed=True)
            dur = int.from_bytes(note[8:], byteorder='big', signed=False)
            print(ivl, dur)


midi = os.listdir('midi')
# random.seed(10)
random.shuffle(midi)
if ".DS_Store" in midi:
    midi.remove(".DS_Store")

t0 = time.perf_counter()
n = 1

for m in midi[:n]:
    print(m)
    seq = midi_to_notes('./midi/' + m)
    durs = seq.T[1]
    # durs_s = sorted(durs, reverse=True)
    # print(durs_s)
    ivls = seq.T[0]
    print(durs.max(), durs.min())
    print(ivls.max(), ivls.min())
    # notes_to_disk(seq)

print((time.perf_counter() - t0) / n)

# notes_to_midi(seq)

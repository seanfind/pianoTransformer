import time
import mido
import os
import numpy as np


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

    # Normalise time values
    res = 1
    avg_dur = np.array([1, (res / np.average(seq, 0)[1])])
    seq = np.multiply(seq, avg_dur)
    seq = np.rint(seq)

    print(np.amax(seq, 0))

    return seq


def notes_to_midi(seq, n0=44, dur=200, upscale=50):
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
        out_track.append(mido.Message('note_on', note=int(n[0]), velocity=int(100 * n[2]), time=int(n[1])))

    midi_out.save('midi_out.mid')


midi = os.listdir('midi')

t0 = time.perf_counter()
n = 100

for m in midi[:n]:
    midi_to_notes('./midi/' + m)

print((time.perf_counter() - t0) / n)

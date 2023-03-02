# import libraries.mido as mido
import mido


def notes_to_midi(seq, dur=200, upscale=200, path='notes_out'):
    # Convert intervals to offsets from a pivot point (currently at 0)
    for i in range(len(seq[:, 0]) - 1, 1, -1):
        seq[:, 0][i] = sum(seq[:, 0][:(i + 1)])

    # Move this central pivot point so that the centre of the offsets is equal to the centre of the range (0, 128)
    centre = (128 // 2) - (seq[:, 0].max() + seq[:, 0].min()) // 2
    seq[:, 0] += centre

    # Make time values absolute
    seq = seq.tolist()
    current_time = 0

    for i in range(len(seq)):
        time_val = seq[i][1]
        seq[i][1] += current_time
        current_time += time_val
        seq[i][1] *= upscale

        seq[i].append(1)  # note_on indicator
        seq.append([seq[i][0], seq[i][1] + dur, 0])

    seq.sort(key=lambda x: x[1])

    for i in range(len(seq) - 1, -1, -1):
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

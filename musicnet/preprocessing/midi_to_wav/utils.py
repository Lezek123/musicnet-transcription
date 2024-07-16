from musicnet.utils import Track, PROJECT_ROOT_DIR
import os
import mido
from mido import MidiFile, MidiTrack
import numpy as np
from glob import glob

BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "preprocessed", "midi_to_wav")
EXTRACTED_MIDI_OUT_DIR = os.path.join(BASE_DIR, "extracted")
CONVERTED_MIDI_OUT_DIR = os.path.join(BASE_DIR, "converted")

# Overrides "get_wav_path" and "get_notes" of Track class to allow easy preprocessing
class MidiConvertedTrack(Track):
    def get_wav_path(self):
        return os.path.join(
            CONVERTED_MIDI_OUT_DIR,
            self.ds,
            f"{self._id}.wav"
        )
    
    def get_notes(self):
        return self.get_norm_midi_notes()

def extract_midi_programs(midi: mido.MidiFile, programs_whitelist: list[int]):
    programs = np.ones(shape=(16), dtype=np.int8) * -1
    channels_whitelist = set()

    filtered_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat, charset=midi.charset)
    filtered_track = MidiTrack()
    filtered_midi.tracks.append(filtered_track)

    skipped_time = 0

    def append_to_filtered(msg):
        nonlocal skipped_time
        filtered_track.append(msg.copy(time=msg.time + skipped_time))
        skipped_time = 0

    p_in = set()
    p_out = set()
    for msg in midi.merged_track:
        if msg.is_meta:
            append_to_filtered(msg)
        elif msg.type == "program_change":
            p_old = programs[msg.channel]
            p_new = int(msg.program)
            programs[msg.channel] = p_new
            p_in.add(p_new)
            if p_old in programs_whitelist:
                channels_whitelist.remove(msg.channel)
            if p_new in programs_whitelist:
                channels_whitelist.add(msg.channel)
                p_out.add(p_new)
            append_to_filtered(msg)
        elif msg.type == "sysex":
            append_to_filtered(msg)
        elif msg.channel in channels_whitelist:
            append_to_filtered(msg)
        else:
            skipped_time += msg.time

    return filtered_midi, p_in, p_out


def list_track_ids(ds_type="train"):
    return list(map(
        lambda f: int(f.split("/")[-1].split('.')[0]),
        glob(os.path.join(CONVERTED_MIDI_OUT_DIR, f"{ds_type}", "*.wav"))
    ))

get_midi_train_ids = lambda: list_track_ids("train")
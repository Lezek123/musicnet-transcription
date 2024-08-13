from dataclasses import dataclass
import pandas as pd
import mido
import numpy as np
import librosa

class BrokenMidiException(Exception): pass

class BaseTrack:
    def get_notes(self) -> pd.DataFrame: raise NotImplementedError()

    def get_wav_path(self) -> str: raise NotImplementedError()

    def get_midi_path(self) -> str: raise NotImplementedError()

    def read_sr(self):
        return librosa.get_samplerate(self.get_wav_path())

    def read_wav_data(self, sr=None):
        return librosa.load(self.get_wav_path(), sr=sr)[0]
    
    def get_duration(self):
        return librosa.get_duration(path=self.get_wav_path())

    def read_midi_file(self):
        try:
            mid = mido.MidiFile(self.get_midi_path())
        except:
            raise BrokenMidiException("Track's midi file is broken or doesn't exist")
        return mid

    def get_midi_notes(self):
        mid = self.read_midi_file()
        notes = []
        note_start_times = np.ones(shape=(16, 128)) * -1
        note_velocities = np.ones(shape=(16, 128)) * -1
        programs = np.ones(shape=(16), dtype=np.int8) * -1
        curr_time = 0
        for msg in mid:
            if msg.is_meta:
                continue
            if msg.type == "program_change":
                programs[msg.channel] = int(msg.program)
            curr_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_start_times[msg.channel][msg.note] = curr_time
                note_velocities[msg.channel][msg.note] = msg.velocity
            if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                # TODO: Inspect why this happens pretty often
                if note_start_times[msg.channel][msg.note] == -1:
                    # print("WARNING! Empty note found!")
                    continue
                notes.append({
                    "note": msg.note,
                    "channel": msg.channel,
                    "instrument": programs[msg.channel] + 1,
                    "start": note_start_times[msg.channel][msg.note],
                    "end": curr_time,
                    "velocity": note_velocities[msg.channel][msg.note]
                })
                note_start_times[msg.channel][msg.note] = -1
                note_velocities[msg.channel][msg.note] = -1
        return pd.DataFrame(notes)
    
@dataclass
class BaseDataset:
    instruments_vocab: dict[int, int]
    notes_vocab: dict[int, int]
    def get_track_ids(self) -> list[int]: raise NotImplementedError()
    def get_track(self, id: int) -> BaseTrack: raise NotImplementedError()
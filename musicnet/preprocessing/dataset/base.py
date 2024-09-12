from dataclasses import dataclass
from musicnet.utils import PROJECT_ROOT_DIR, IS_CLOUD
from musicnet.config.dataset.DatasetConfig import DsConfig, DsConfigOverrides
from glob import glob
from typing import Optional, Literal, TypeAlias
import pandas as pd
import mido
import numpy as np
import librosa
import os

if IS_CLOUD:
    DATA_PATH = "/gcs/musicnet-job-data/data"
else:
    DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data")
    

DATA_SOURCES_PATH = os.path.join(DATA_PATH, "sources")
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")

DsName: TypeAlias = Literal["train", "val", "test"]

@dataclass
class DsInfo:
    name: DsName
    config: DsConfig

    def __init__(self, ds_name: DsName, default_config: DsConfig, overrides: Optional[DsConfigOverrides]):
        self.name = ds_name
        self.src_name = ds_name if (overrides and overrides.wav_source) else "default"
        self.config = default_config if overrides is None else default_config.with_overrides(overrides)

    def get_source_path(self):
        return os.path.join(DATA_SOURCES_PATH, self.src_name)
    
    def get_preprocessed_path(self):
        return os.path.join(PREPROCESSED_DATA_PATH, self.name)

class BrokenMidiException(Exception): pass

@dataclass
class BaseTrack:
    _id: int
    ds_name: str

    def get_track_ids(self) -> list[int]:
        return list(map(
            lambda f: int(f.split("/")[-1].split('.')[0]),
            glob(os.path.join(DATA_SOURCES_PATH, self.ds_name, "*.wav"))
        ))
    
    def get_path_with_ext(self, ext: str) -> str:
        return os.path.join(
            DATA_SOURCES_PATH,
            self.ds_name,
            f"{self._id}.{ext}"
        )

    def get_wav_path(self) -> str:
        return self.get_path_with_ext("wav")
    
    def get_midi_path(self) -> str:
        return self.get_path_with_ext("midi")

    def get_notes(self) -> pd.DataFrame: raise NotImplementedError()

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
            curr_time += msg.time
            if msg.is_meta:
                continue
            if msg.type == "program_change":
                programs[msg.channel] = int(msg.program)
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
    name: str
    instruments: set[int]
    notes: set[int]

    def get_track_ids(self) -> list[int]:
        return list(map(
            lambda f: int(f.split("/")[-1].split('.')[0]),
            glob(os.path.join(DATA_SOURCES_PATH, self.name, "*.wav"))
        ))

    def get_track(self, id: int) -> BaseTrack: raise NotImplementedError()
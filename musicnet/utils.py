import os
from glob import glob
import pandas as pd
import mido
import numpy as np
import librosa
from pathlib import Path
import yaml
from functools import reduce
from datetime import datetime
import operator

PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)
if os.environ.get("CLOUD_ML_PROJECT_ID"):
    DATASET_BASE_PATH = "/gcs/musicnet-ds/MusicNet"
else:
    DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "MusicNet")
DATASET_XY_PATH = os.path.join(DATASET_BASE_PATH, "musicnet", "musicnet")

def create_vocab(value_set):
    values = list(value_set)
    values.sort()
    return { value: index for index, value in enumerate(values) }

def load_instruments_and_notes():
    instruments = set()
    notes = set()
    for file in glob(os.path.join(DATASET_XY_PATH, "train_labels", "*.csv")):
        data = pd.read_csv(file)
        for i in data["instrument"].unique():
            instruments.add(i)
        for n in data["note"].unique():
            notes.add(n)

    return create_vocab(instruments), create_vocab(notes)

def list_track_ids(ds_type="train"):
    return list(map(
        lambda f: int(f.split("/")[-1].split('.')[0]),
        glob(os.path.join(DATASET_XY_PATH, f"{ds_type}_labels", "*.csv"))
    ))

metadata = pd.read_csv(os.path.join(DATASET_BASE_PATH, "musicnet_metadata.csv")).set_index("id")
instruments_vocab, notes_vocab = load_instruments_and_notes()
get_train_ids = lambda: list_track_ids("train")
get_test_ids = lambda: list_track_ids("test")

class BrokenMidiException(Exception): pass

class Track:
    def __init__(self, _id):
        self._id = _id
        if os.path.exists(os.path.join(DATASET_XY_PATH, "train_data", f"{self._id}.wav")):
            self.ds = "train"
        elif os.path.exists((os.path.join(DATASET_XY_PATH, "test_data", f"{self._id}.wav"))):
            self.ds = "test"
        else:
            raise Exception(f"Track {_id} not found!")
        self.metadata = metadata.loc[_id]
    
    def get_wav_path(self):
        return os.path.join(DATASET_XY_PATH, f"{self.ds}_data", f"{self._id}.wav")
    
    def get_csv_path(self):
        return os.path.join(DATASET_XY_PATH, f"{self.ds}_labels", f"{self._id}.csv")
    
    def get_metadata(self):
        return self.metadata
    
    def get_midi_path(self):
        pattern = os.path.join(
            DATASET_BASE_PATH,
            "musicnet_midis",
            "musicnet_midis",
            self.metadata["composer"],
            f"{self._id}_*.mid"
        )
        found = glob(pattern)
        if len(found):
            return found[0]
        else:
            raise Exception(f"Track {self._id} midi file not found!")
        
    def read_csv_data(self):
        return pd.read_csv(self.get_csv_path())
    
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
            raise BrokenMidiException("Track's midi file is broken")
        return mid

    def read_midi_notes(self):
        mid = self.read_midi_file()
        notes = []
        note_start_times = np.ones(shape=(16, 128)) * -1
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
            if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                # TODO: Inspect why this happens pretty often
                # if note_start_times[msg.channel][msg.note] == -1:
                #     print("WARNING! Empty note found!")
                notes.append({
                    "note": msg.note,
                    "channel": msg.channel,
                    "program": programs[msg.channel],
                    "start_time": note_start_times[msg.channel][msg.note],
                    "end_time": curr_time
                })
                note_start_times[msg.channel][msg.note] = -1
        return pd.DataFrame(notes)
    
    def get_norm_csv_notes(self):
        sr = self.read_sr()
        csv_data = self.read_csv_data()
        notes = pd.DataFrame({
            "start": csv_data["start_time"] / sr,
            "end": csv_data["end_time"] / sr,
            "instrument": csv_data["instrument"],
            "note": csv_data["note"]
        })
        return notes
    
    def get_norm_midi_notes(self):
        midi_notes = self.read_midi_notes()
        notes = pd.DataFrame({
            "start": midi_notes["start_time"],
            "end": midi_notes["end_time"],
            "instrument": midi_notes["program"] + 1,
            "note": midi_notes["note"]
        })
        return notes
    
    def get_notes(self):
        return self.get_norm_csv_notes()
    
def load_params(paths=None):
    all_params = yaml.safe_load(open(os.path.join(PROJECT_ROOT_DIR, "params.yaml")))
    if paths:
        params = dict()
        for path in paths:
            subpaths = path.split(".")
            if subpaths[-1] == "*":
                value = reduce(operator.getitem, subpaths[:-1], all_params)
                params = { **params, **value }
            else:
                value = reduce(operator.getitem, subpaths, all_params)                
                params[subpaths[-1]] = value
        return params
    return all_params

def note_frequency(note_idx):
    note = list(notes_vocab.keys())[note_idx]
    return 440 * (2 ** ((note - 69) / 12))

def get_training_artifacts_dir(script_path: Path):
    if os.environ.get("CLOUD_ML_PROJECT_ID"):
        date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return f"/gcs/musicnet-ds/jobs/{date_str}"
    else:
        return os.path.dirname(script_path)


from musicnet.utils import PROJECT_ROOT_DIR, IS_CLOUD, create_vocab
from .base import BaseDataset, BaseTrack
from dataclasses import dataclass
from glob import glob
import os
import pandas as pd

if IS_CLOUD:
    DATASET_BASE_PATH = "/gcs/musicnet-ds/MusicNet"
else:
    DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "MusicNet")

DATASET_XY_PATH = os.path.join(DATASET_BASE_PATH, "musicnet", "musicnet")
DATASET_MIDIS_PATH = os.path.join(DATASET_BASE_PATH, "musicnet_midis", "musicnet_midis")

class MusicNetTrack(BaseTrack):
    def __init__(self, _id: int, metadata: pd.DataFrame, ds_type="train"):
        self.ds_type = ds_type
        self._id = _id
        self.metadata = metadata.loc[_id]
    
    def get_wav_path(self):
        return os.path.join(DATASET_XY_PATH, f"{self.ds_type}_data", f"{self._id}.wav")
    
    def get_midi_path(self):
        pattern = os.path.join(
            DATASET_MIDIS_PATH,
            str(self.metadata["composer"]),
            f"{self._id}_*.mid"
        )
        found = glob(pattern)
        if len(found):
            return found[0]
        else:
            raise Exception(f"Track {self._id} midi file not found!")

    def get_csv_path(self):
        return os.path.join(DATASET_XY_PATH, f"{self.ds_type}_labels", f"{self._id}.csv")
    
    def get_metadata(self):
        return self.metadata
        
    def read_csv_data(self):
        return pd.read_csv(self.get_csv_path())
    
    def get_csv_notes(self):
        sr = self.read_sr()
        csv_data = self.read_csv_data()
        notes = pd.DataFrame({
            "start": csv_data["start_time"] / sr,
            "end": csv_data["end_time"] / sr,
            "instrument": csv_data["instrument"],
            "note": csv_data["note"]
        })
        return notes
    
    def get_notes(self):
        return self.get_csv_notes()
    
@dataclass
class MusicNetDataset(BaseDataset):
    def __init__(self):
        self.metadata = pd.read_csv(os.path.join(DATASET_BASE_PATH, "musicnet_metadata.csv")).set_index("id")
        instruments_vocab, notes_vocab = self.load_instruments_and_notes()
        self.instruments_vocab = instruments_vocab
        self.notes_vocab = notes_vocab
        super().__init__(instruments_vocab, notes_vocab)

    def load_instruments_and_notes(self):
        instruments = set()
        notes = set()
        for file in glob(os.path.join(DATASET_XY_PATH, "train_labels", "*.csv")):
            data = pd.read_csv(file)
            for i in data["instrument"].unique():
                instruments.add(i)
            for n in data["note"].unique():
                notes.add(n)

        return create_vocab(instruments), create_vocab(notes)

    def get_track_ids(self, ds_type="train") -> list[int]:
        return list(map(
            lambda f: int(f.split("/")[-1].split('.')[0]),
            glob(os.path.join(DATASET_XY_PATH, f"{ds_type}_labels", "*.csv"))
        ))
    
    def get_track(self, id: int, ds_type="train"):
        return MusicNetTrack(id, self.metadata, ds_type)
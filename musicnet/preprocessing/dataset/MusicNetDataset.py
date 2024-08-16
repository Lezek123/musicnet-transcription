from musicnet.utils import PROJECT_ROOT_DIR, IS_CLOUD
from musicnet.config.dataset.wav_source.common import MnDatasetType
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
    def __init__(self, _id: int, ds_name: str, metadata: pd.DataFrame, mn_ds_type: MnDatasetType):
        self.mn_ds_type = mn_ds_type
        self._id = _id
        self.metadata = metadata.loc[_id]
        super().__init__(_id, ds_name)
    
    def get_wav_path(self) -> str:
        return os.path.join(DATASET_XY_PATH, f"{self.mn_ds_type.value}_data", f"{self._id}.wav")
    
    def get_midi_path(self) -> str:
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

    def get_csv_path(self) -> str:
        return os.path.join(DATASET_XY_PATH, f"{self.mn_ds_type.value}_labels", f"{self._id}.csv")
    
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
    def __init__(self, name: str, mn_ds_type: MnDatasetType):
        self.mn_ds_type = mn_ds_type
        self.metadata = pd.read_csv(os.path.join(DATASET_BASE_PATH, "musicnet_metadata.csv")).set_index("id")
        instruments, notes = self.load_instruments_and_notes()
        super().__init__(name, instruments, notes)

    def load_instruments_and_notes(self):
        instruments = set()
        notes = set()
        for file in glob(os.path.join(DATASET_XY_PATH, "train_labels", "*.csv")):
            data = pd.read_csv(file)
            for i in data["instrument"].unique():
                instruments.add(i)
            for n in data["note"].unique():
                notes.add(n)

        return instruments, notes

    def get_track_ids(self) -> list[int]:
        return list(map(
            lambda f: int(f.split("/")[-1].split('.')[0]),
            glob(os.path.join(DATASET_XY_PATH, f"{self.mn_ds_type.value}_labels", "*.csv"))
        ))
    
    def get_track(self, id: int):
        return MusicNetTrack(id, self.name, self.metadata, self.mn_ds_type)
from musicnet.utils import PROJECT_ROOT_DIR
from .MusicNetDataset import MusicNetDataset, MusicNetTrack
from .ConvertableMidiTrack import ConvertableMidiTrack
from dataclasses import dataclass
from glob import glob
import pandas as pd
import os

DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "preprocessed", "midi_to_wav")
DATASET_EXTRACTED_MIDI_PATH = os.path.join(DATASET_BASE_PATH, "extracted")
DATASET_WAVS_PATH = os.path.join(DATASET_BASE_PATH, "converted")

class MusicNetMidiConvertedTrack(MusicNetTrack, ConvertableMidiTrack):
    def __init__(self, _id: int, metadata: pd.DataFrame, is_extracted: bool, ds_type="train"):
        self.is_extracted = is_extracted
        super().__init__(_id, metadata, ds_type)
        
    def get_wav_path(self):
        return os.path.join(
            DATASET_WAVS_PATH,
            self.ds_type,
            f"{self._id}.wav"
        )
    
    def get_midi_path(self):
        extracted_midi_path = os.path.join(DATASET_EXTRACTED_MIDI_PATH, f"{self._id}.midi")
        if self.is_extracted:
            if os.path.exists(extracted_midi_path):
                return extracted_midi_path
            else:
                raise Exception("Missing extracted MIDI")
        return super().get_midi_path()
    
    def get_notes(self):
        return self.get_midi_notes()
    
@dataclass
class MusicNetMidiConvertedDataset(MusicNetDataset):
    def __init__(self, programs_whitelist: list[int]):
        super().__init__()
        self.programs_whitelist = programs_whitelist
        self.instruments_vocab = { (v+1): k for k, v in enumerate(programs_whitelist) }

    def get_track_ids(self, ds_type="train") -> list[int]:
        return list(map(
            lambda f: int(f.split("/")[-1].split('.')[0]),
            glob(os.path.join(DATASET_WAVS_PATH, f"{ds_type}", "*.wav"))
        ))

    def get_track(self, id: int, ds_type="train"):
        return MusicNetMidiConvertedTrack(id, self.metadata, bool(self.programs_whitelist), ds_type)
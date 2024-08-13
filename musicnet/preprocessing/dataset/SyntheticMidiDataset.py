from .base import BaseDataset
import os
import pandas as pd
from musicnet.utils import PROJECT_ROOT_DIR, create_vocab
from glob import glob
from musicnet.config.dataset.wav_source.SynthMidiToWavConfig import SynthMidiToWavConfig
from dataclasses import dataclass
from .ConvertableMidiTrack import ConvertableMidiTrack

DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "preprocessed", "synth_midi")
DATASET_WAVS_PATH = os.path.join(DATASET_BASE_PATH, "wav")
DATASET_MIDIS_PATH = os.path.join(DATASET_BASE_PATH, "midi")

class SyntheticTrack(ConvertableMidiTrack):
    def __init__(self, id: int):
        self._id = id
    
    def get_wav_path(self):
        return os.path.join(DATASET_WAVS_PATH, f"{self._id}.wav")
    
    def get_midi_path(self):
        return os.path.join(DATASET_MIDIS_PATH, f"{self._id}.midi")
    
    def get_notes(self) -> pd.DataFrame:
        return self.get_midi_notes()

@dataclass  
class SyntheticMidiDataset(BaseDataset):
    def __init__(self, config: SynthMidiToWavConfig):
        # Only one instrument allowed for now
        instruments_vocab = create_vocab([0])
        notes_vocab = create_vocab(list(range(config.min_note, config.max_note)))
        super().__init__(instruments_vocab, notes_vocab)

    def get_track_ids(self) -> list[int]:
        return list(map(
            lambda f: int(f.split("/")[-1].split('.')[0]),
            glob(os.path.join(DATASET_WAVS_PATH, "*.wav"))
        ))
    
    def get_track(self, id: int):
        return SyntheticTrack(id)
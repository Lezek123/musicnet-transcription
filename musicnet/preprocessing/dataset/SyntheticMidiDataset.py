from .base import BaseDataset
import pandas as pd
from musicnet.config.dataset.wav_source.SynthMidiToWavConfig import SynthMidiToWavConfig
from dataclasses import dataclass
from .ConvertableMidiTrack import ConvertableMidiTrack

class SyntheticTrack(ConvertableMidiTrack):
    def get_metadata_path(self) -> str:
        return self.get_path_with_ext(".csv")

    def get_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.get_metadata_path(), index_col=0)

    def get_notes(self) -> pd.DataFrame:
        return self.get_midi_notes()


@dataclass  
class SyntheticMidiDataset(BaseDataset):
    def __init__(self, name: str, config: SynthMidiToWavConfig):
        # Only one instrument allowed for now
        instruments = set([1])
        notes = set(range(config.min_note, config.max_note))
        super().__init__(name, instruments, notes)
    
    def get_track(self, id: int):
        return SyntheticTrack(id, self.name)
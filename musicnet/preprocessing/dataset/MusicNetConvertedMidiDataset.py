from musicnet.config.dataset.wav_source.common import MnDatasetType
from .MusicNetDataset import MusicNetDataset, MusicNetTrack
from .ConvertableMidiTrack import ConvertableMidiTrack
from .base import BaseDataset
from dataclasses import dataclass

class MusicNetMidiConvertedTrack(MusicNetTrack, ConvertableMidiTrack):
    def get_midi_path(self) -> str:
        return self.get_path_with_ext("midi")
    
    def get_wav_path(self) -> str:
        return self.get_path_with_ext("wav")
    
    def get_notes(self):
        return self.get_midi_notes()
    
@dataclass
class MusicNetMidiConvertedDataset(MusicNetDataset):
    def __init__(self, name: str, mn_ds_type: MnDatasetType, programs_whitelist: list[int]):
        super().__init__(name, mn_ds_type)
        self.programs_whitelist = programs_whitelist
        self.instruments_vocab = { (v+1): k for k, v in enumerate(programs_whitelist) }

    def get_track_ids(self) -> list[int]:
        return BaseDataset.get_track_ids(self)

    def get_track(self, id: int):
        return MusicNetMidiConvertedTrack(id, self.name, self.metadata, self.mn_ds_type)
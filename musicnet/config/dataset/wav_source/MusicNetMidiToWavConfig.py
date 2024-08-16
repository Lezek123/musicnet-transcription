from dataclasses import dataclass, field
from .common import WavSource, WavSourceType, MnDatasetType

@dataclass
class MusicNetMidiToWavConfig(WavSource):
    type: WavSourceType = WavSourceType.MUSICNET_MIDI_TO_WAV
    mn_ds_type: MnDatasetType = MnDatasetType.TRAIN
    programs_whitelist: list[int] = field(default_factory=lambda : [0])
from dataclasses import dataclass
from enum import Enum

class WavSourceType(Enum):
    MUSICNET_MIDI_TO_WAV = "musicnet_midi_to_wav"
    MUSICNET_WAV = "musicnet_wav"
    SYNTH_MIDI_TO_WAV = "synth_midi_to_wav"

class MnDatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    
@dataclass
class WavSource:
    type: WavSourceType
from dataclasses import dataclass, field
from .common import WavSource, WavSourceType

@dataclass
class MusicNetMidiToWavConfig(WavSource):
    type: WavSourceType = WavSourceType.MUSICNET_MIDI_TO_WAV
    programs_whitelist: list[int] = field(default_factory=lambda : [0])
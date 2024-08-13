from dataclasses import dataclass
from .common import WavSource, WavSourceType

@dataclass
class SynthMidiToWavConfig(WavSource):
    type: WavSourceType = WavSourceType.SYNTH_MIDI_TO_WAV
    # Ticks per second
    tps: int = 100
    min_note_ticks: int = 1
    max_note_ticks: int = 100
    max_silmultaneous_notes: int = 3
    sidechannel_sparsity: int = 1
    min_note: int = 20
    max_note: int = 105
    # Sidechannel notes will be chosen from standard distribution centered
    # at the main note (but excluding the main note as potential value)
    # with notes_std as standard distribution
    notes_std: int = 10
    track_length_per_note: int = 3600
    velocity_min: int = 64
    velocity_max: int = 127
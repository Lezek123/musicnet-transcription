from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from .common import Preprocessor, PreprocessorType

class SpectogramUnit(Enum):
    AMPLITUDE = "amplitude"
    DECIBELS = "decibels"

@dataclass
class SpectogramParams:
    # Number of samples in a window per fft in spectrogram
    n_fft: int = 2205
    # The amount of samples we are shifting after each fft in spectrogram
    hop_length: int = 441
    # Minimum spectogram frequency in Hz
    min_hz: int = 0
    # Number of filters in the spectrogram (also determines max_hz)
    n_filters: int = 200
    # Spectogram values unit
    unit: SpectogramUnit = SpectogramUnit.DECIBELS

@dataclass
class WavChunksTFRecordPreprocessorParams:
    chunk_size_sec: int = 10
    chunk_shift_sec: int = 5
    target_sr: int = 44100
    note_rounding: float = 0.01
    spectogram: Optional[SpectogramParams] = field(default_factory=lambda : SpectogramParams())

@dataclass
class WavChunksTFRecordDatasetSplit:
    size: int | float
    file_count: int

@dataclass
class WavChunksTFRecordPreprocessorConfig(Preprocessor):
    type: PreprocessorType = PreprocessorType.WAV_CHUNKS_TFRECORDS
    params: WavChunksTFRecordPreprocessorParams = field(default_factory=lambda : WavChunksTFRecordPreprocessorParams())
    ds_split: dict[str, WavChunksTFRecordDatasetSplit] = field(default_factory=lambda : {
        "train": WavChunksTFRecordDatasetSplit(size=0.8, file_count=20),
        "val": WavChunksTFRecordDatasetSplit(size=0.2, file_count=20)
    })
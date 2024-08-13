from dataclasses import dataclass
from omegaconf import MISSING
from .wav_source.common import WavSource
from .preprocessor import Preprocessor

@dataclass
class DatasetConfig:
    wav_source: WavSource = MISSING
    preprocessor: Preprocessor = MISSING
    load_fraction: float = 1.0
    batch_size: int = 8
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, Optional
from .wav_source.common import WavSource
from .preprocessor import Preprocessor

@dataclass
class DsConfigOverrides:
    wav_source: Optional[WavSource] = None
    batch_size: Optional[int] = None
    split: Optional[Tuple[float, float]] = None

@dataclass
class DsConfig:
    wav_source: WavSource = MISSING
    preprocessor: Preprocessor = MISSING
    batch_size: int = 8
    split: Optional[Tuple[float, float]] = None

    def with_overrides(self, overrides: DsConfigOverrides):
        return DsConfig(
            wav_source=self.wav_source if overrides.wav_source is None else overrides.wav_source,
            preprocessor=self.preprocessor,
            batch_size=self.batch_size if overrides.batch_size is None else overrides.batch_size,
            split=self.split if overrides.split is None else overrides.split
        )

@dataclass
class DatasetConfig:
    default: DsConfig = MISSING
    train: Optional[DsConfigOverrides] = None
    val: Optional[DsConfigOverrides] = None
    test: Optional[DsConfigOverrides] = None
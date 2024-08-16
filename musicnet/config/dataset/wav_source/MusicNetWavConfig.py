from dataclasses import dataclass
from .common import WavSource, MnDatasetType, WavSourceType

@dataclass
class MusicNetWavConfig(WavSource):
    type: WavSourceType = WavSourceType.MUSICNET_WAV
    mn_ds_type: MnDatasetType = MnDatasetType.TRAIN
from dataclasses import dataclass, field
from .dataset.DatasetConfig import DatasetConfig
from .model import Model, CNNConfig, TransformerConfig, WaveNetConfig
from .dataset.wav_source import MusicNetMidiToWavConfig, MusicNetWavConfig, SynthMidiToWavConfig
from .dataset.preprocessor import WavChunksTFRecordPreprocessorConfig
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any

defaults = [
    "_self_",
    {"model": "cnn"},
    {"dataset/default/wav_source": "synth_midi_to_wav"},
    {"dataset/default/preprocessor": "wav_chunks_tfrecord"}
]

@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    dataset: DatasetConfig = field(default_factory=lambda : DatasetConfig())
    model: Model = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

cs.store(group="model", name="cnn", node=CNNConfig)
cs.store(group="model", name="transformer", node=TransformerConfig)
cs.store(group="model", name="wavenet", node=WaveNetConfig)

# Preprocessors
cs.store(group=f"dataset/default/preprocessor", name="wav_chunks_tfrecord", node=WavChunksTFRecordPreprocessorConfig)

for ds_type in ["default", "train", "val", "test"]:
    # Wav sources
    cs.store(group=f"dataset/{ds_type}/wav_source", name="music_net_midi_to_wav", node=MusicNetMidiToWavConfig)
    cs.store(group=f"dataset/{ds_type}/wav_source", name="music_net_wav", node=MusicNetWavConfig)
    cs.store(group=f"dataset/{ds_type}/wav_source", name="synth_midi_to_wav", node=SynthMidiToWavConfig)
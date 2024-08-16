from typing import Optional
from musicnet.utils import create_vocab
from musicnet.config.Config import Config
from musicnet.config.dataset.wav_source import MusicNetWavConfig, SynthMidiToWavConfig, MusicNetMidiToWavConfig
from musicnet.preprocessing.dataset import MusicNetDataset, MusicNetMidiConvertedDataset, SyntheticMidiDataset, DsInfo
from musicnet.config.dataset.DatasetConfig import DsConfig, DsConfigOverrides
from musicnet.preprocessing.dataset.base import DsName

def get_datasets_info(config: Config) -> list[DsInfo]:
    default_config = config.dataset.default
    ds_infos = []
    ds_names: list[DsName] = ["train", "val", "test"]
    for ds_name in ds_names:
        overrides: Optional[DsConfigOverrides] = config.dataset.__getattribute__(ds_name)
        ds_info = DsInfo(ds_name, default_config, overrides)
        ds_infos.append(ds_info)
    return ds_infos

def load_vocabs(config: Config):
    instruments: set[int] = set()
    notes: set[int] = set()
    ds_infos = get_datasets_info(config)
    for ds_info in ds_infos:
        src_ds = load_source_dataset(ds_info.config, ds_info.src_name)
        instruments = instruments | set(src_ds.instruments)
        notes = notes | set(src_ds.notes)
    return create_vocab(instruments), create_vocab(notes)

def load_source_dataset(config: DsConfig, name: str):
    wav_source_config = config.wav_source
    if isinstance(wav_source_config, MusicNetMidiToWavConfig):
        return MusicNetMidiConvertedDataset(name, wav_source_config.mn_ds_type, wav_source_config.programs_whitelist)
    if isinstance(wav_source_config, MusicNetWavConfig):
        return MusicNetDataset(name, wav_source_config.mn_ds_type)
    if isinstance(wav_source_config, SynthMidiToWavConfig):
        return SyntheticMidiDataset(name, wav_source_config)
    else:
        raise Exception("Unrecognized wav_source")
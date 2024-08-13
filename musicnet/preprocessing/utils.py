from musicnet.config.Config import Config
from musicnet.config.dataset.wav_source import MusicNetWavConfig, SynthMidiToWavConfig, MusicNetMidiToWavConfig
from musicnet.preprocessing.dataset import MusicNetDataset, MusicNetMidiConvertedDataset, SyntheticMidiDataset

def load_dataset(config: Config):
    wav_source_config = config.dataset.wav_source
    if isinstance(wav_source_config, MusicNetMidiToWavConfig):
        return MusicNetMidiConvertedDataset(wav_source_config.programs_whitelist)
    if isinstance(wav_source_config, MusicNetWavConfig):
        return MusicNetDataset()
    if isinstance(wav_source_config, SynthMidiToWavConfig):
        return SyntheticMidiDataset(wav_source_config)
    else:
        raise Exception("Unrecognized wav_source")
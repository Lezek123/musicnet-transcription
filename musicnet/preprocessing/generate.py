import hydra
import typing
from musicnet.config.Config import Config
from omegaconf import OmegaConf
from musicnet.config.dataset.wav_source import MusicNetWavConfig, MusicNetMidiToWavConfig, SynthMidiToWavConfig
from .midi_to_wav.convert import convert as convert_mn_midi_to_wav
from .synth_midi.generate import generate as generate_synth_midi

@hydra.main(version_base=None, config_name="config")
def generate(cfg: Config):
    config = typing.cast(Config, OmegaConf.to_object(cfg))
    wav_source_cfg = config.dataset.wav_source
    
    if isinstance(wav_source_cfg, MusicNetMidiToWavConfig):
        convert_mn_midi_to_wav(wav_source_cfg)
    elif isinstance(wav_source_cfg, SynthMidiToWavConfig):
        generate_synth_midi(wav_source_cfg)
    elif isinstance(wav_source_cfg, MusicNetWavConfig):
        print("Nothing to do, skipping...")
    else:
        raise Exception("Unrecognized wav_source configuration!")
    
generate()
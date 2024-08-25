import os
from musicnet.config import to_config_object, Config
from musicnet.config.dataset.wav_source import MusicNetWavConfig, MusicNetMidiToWavConfig, SynthMidiToWavConfig
from .midi_to_wav.convert import convert as convert_mn_midi_to_wav
from .synth_midi.generate import generate as generate_synth_midi
from .utils import get_datasets_info
from musicnet.utils import recreate_dirs
from .dataset.base import DATA_SOURCES_PATH

def generate(cfg: Config):
    config = to_config_object(cfg)
    datasets_info = get_datasets_info(config)

    recreate_dirs([DATA_SOURCES_PATH])

    for ds_info in datasets_info:
        if not os.path.exists(ds_info.get_source_path()):
            print(f"Generating {ds_info.src_name} source dataset using {ds_info.config.wav_source.type.value}...\n")
            if isinstance(ds_info.config.wav_source, MusicNetMidiToWavConfig):
                convert_mn_midi_to_wav(ds_info.config.wav_source, ds_info.src_name)
            elif isinstance(ds_info.config.wav_source, SynthMidiToWavConfig):
                generate_synth_midi(ds_info.config.wav_source, ds_info.src_name)
            elif isinstance(ds_info.config.wav_source, MusicNetWavConfig):
                # TODO: Copy files?
                print("Nothing to do, skipping...")
            else:
                raise Exception("Unrecognized wav_source configuration!")
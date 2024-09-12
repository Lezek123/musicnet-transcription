import os
from musicnet.config import Config
from musicnet.config.dataset.wav_source import MusicNetWavConfig, MusicNetMidiToWavConfig, SynthMidiToWavConfig
from .midi_to_wav.convert import convert as convert_mn_midi_to_wav
from .synth_midi.generate import generate as generate_synth_midi
from .utils import get_datasets_info
from musicnet.utils import recreate_dirs
from musicnet.PipelineState import PipelineState, StageState
from .dataset.base import DATA_SOURCES_PATH

def generate(config: Config, ps: PipelineState):
    datasets_info = get_datasets_info(config)

    if ps.stage_state == StageState.CLEAN:
        recreate_dirs(list(set([os.path.join(DATA_SOURCES_PATH, dsi.src_name) for dsi in datasets_info])))
    ps.set_stage_state(StageState.IN_PROGRESS)

    covered = set()
    for ds_info in datasets_info:
        if ds_info.src_name in covered:
            continue
        covered.add(ds_info.src_name)
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
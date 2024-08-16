from .utils import extract_midi_programs
from multiprocessing import Pool
import os
import shutil
from musicnet.utils import recreate_dirs
from musicnet.config.dataset.wav_source.MusicNetMidiToWavConfig import MusicNetMidiToWavConfig
from ..dataset.MusicNetDataset import MusicNetDataset
from ..dataset.MusicNetConvertedMidiDataset import MusicNetMidiConvertedTrack
from ..dataset.base import BrokenMidiException, DATA_SOURCES_PATH
from functools import partial

def prepare_track(id: int, ds_name: str, config: MusicNetMidiToWavConfig):
    mn_dataset = MusicNetDataset(ds_name, config.mn_ds_type)
    out_dir = os.path.join(DATA_SOURCES_PATH, ds_name)
    track = mn_dataset.get_track(id)
    try:
        midi = track.read_midi_file()
    except BrokenMidiException:
        print("TRACK NOT CONVERTED: BROKEN MIDI FILE", id)
        return -1
    
    out_midi_path = os.path.join(out_dir, f"{id}.midi")
    if config.programs_whitelist:
        filtered_midi, p_in, p_out = extract_midi_programs(midi, config.programs_whitelist)
        if len(p_out) == 0:
            print("TRACK NOT CONVERTED: NO OUTPUT PROGRAMS PRESENT", id)
            return -1
        filtered_midi.save(out_midi_path)
        print(f"Track {id} midi programs successfully extracted! p_in: {len(p_in)}, p_out: {len(p_out)}")
        return id
    else:
        shutil.copyfile(track.get_midi_path(), out_midi_path)
        return id
    
def convert_track(id: int, ds_name: str, config: MusicNetMidiToWavConfig):
    mn_dataset = MusicNetDataset(ds_name, config.mn_ds_type)
    track = MusicNetMidiConvertedTrack(id, ds_name, mn_dataset.metadata, config.mn_ds_type)
    track.generate_wav()
    print(f"Track {id} successfully converted to wav...")

def convert(config: MusicNetMidiToWavConfig, ds_name: str):
    mn_dataset = MusicNetDataset(ds_name, config.mn_ds_type)
    track_ids = mn_dataset.get_track_ids()
    out_dir = os.path.join(DATA_SOURCES_PATH, ds_name)
    recreate_dirs([out_dir])

    prepare_track_partial = partial(prepare_track, ds_name=ds_name, config=config)
    convert_track_partial = partial(convert_track, ds_name=ds_name, config=config)
    with Pool(processes=8) as pool:
        track_ids = pool.map(prepare_track_partial, track_ids)
        track_ids = list(filter(lambda id: id != -1, track_ids))
        pool.map(convert_track_partial, track_ids)
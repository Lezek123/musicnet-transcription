from .utils import extract_midi_programs
from multiprocessing import Pool
import os
from musicnet.utils import recreate_dirs
from musicnet.config.dataset.wav_source.MusicNetMidiToWavConfig import MusicNetMidiToWavConfig
from ..dataset.MusicNetDataset import MusicNetDataset
from ..dataset.MusicNetConvertedMidiDataset import DATASET_EXTRACTED_MIDI_PATH, DATASET_WAVS_PATH, MusicNetMidiConvertedTrack
from ..dataset.base import BrokenMidiException

def convert(config: MusicNetMidiToWavConfig):
    mn_dataset = MusicNetDataset()

    datasets = {
        "train": mn_dataset.get_track_ids("train"),
        "test": mn_dataset.get_track_ids("test")
    }

    recreate_dirs([os.path.join(DATASET_WAVS_PATH, ds_name) for ds_name in datasets.keys()])

    if config.programs_whitelist:
        recreate_dirs([DATASET_EXTRACTED_MIDI_PATH])

    def prepare_track(id, ds_type):
        track = mn_dataset.get_track(id, ds_type=ds_type)
        try:
            midi = track.read_midi_file()
        except BrokenMidiException:
            print("TRACK NOT CONVERTED: BROKEN MIDI FILE", id)
            return None
        if config.programs_whitelist:
            filtered_midi, p_in, p_out = extract_midi_programs(midi, config.programs_whitelist)
            if len(p_out) == 0:
                print("TRACK NOT CONVERTED: NO OUTPUT PROGRAMS PRESENT", id)
                return None
            midi_path = os.path.join(DATASET_EXTRACTED_MIDI_PATH, f"{id}.midi")
            filtered_midi.save(midi_path)
            print(f"Track {id} midi programs successfully extracted! p_in: {len(p_in)}, p_out: {len(p_out)}")
            return id
        else:
            return id

    def convert_track(id, ds_type):
        track = MusicNetMidiConvertedTrack(id, mn_dataset.metadata, ds_type)
        track.generate_wav()
        print(f"Track {id} successfully converted to wav...")

    with Pool(processes=8) as pool:
        for ds_name, ds_ids in datasets.items():
            track_ids = pool.map(lambda id: prepare_track(id, ds_name), ds_ids)
            track_ids = list(filter(lambda id: id, track_ids))
            pool.starmap(convert_track, [(id, ds_name) for id in track_ids])
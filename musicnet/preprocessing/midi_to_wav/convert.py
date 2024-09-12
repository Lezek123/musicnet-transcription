from .utils import extract_midi_programs
from multiprocessing import Pool, Manager
import os
import shutil
import numpy as np
from musicnet.config.dataset.wav_source.MusicNetMidiToWavConfig import MusicNetMidiToWavConfig
from ..dataset.MusicNetDataset import MusicNetDataset
from ..dataset.MusicNetConvertedMidiDataset import MusicNetMidiConvertedTrack
from ..dataset.base import BrokenMidiException, DATA_SOURCES_PATH

class State():
    def __init__(self, ds_name: str, lock):
        self.prepared_state_path = os.path.join(DATA_SOURCES_PATH, ds_name, "prepared")
        self.converted_state_path = os.path.join(DATA_SOURCES_PATH, ds_name, "converted")
        self.lock = lock

    def add_converted_file(self, id: int):
        with self.lock:
            with open(self.converted_state_path, "a") as converted_state_file:
                converted_state_file.write(f"{id}\n")

    def add_prepared_file(self, id: int, result: bool):
        with self.lock:
            with open(self.prepared_state_path, "a") as prepared_state_file:
                prepared_state_file.write(f"{id}:{result}\n")

    def get_prepared_ids_and_results(self):
        ids = []
        results = []
        if os.path.exists(self.prepared_state_path):
            with open(self.prepared_state_path, "r") as prepared_state_file:
                for l in prepared_state_file.readlines():
                    if l:
                        id, result = l.split(":")
                        ids.append(int(id))
                        results.append(result.strip() == "True")
        return ids, results
    
    def get_converted_ids(self):
        ids = []
        if os.path.exists(self.converted_state_path):
            with open(self.converted_state_path, "r") as converted_state_file:
                for l in converted_state_file.readlines():
                    if l:
                        ids.append(int(l))
        return ids


def prepare_track(id: int, lock, ds_name: str, config: MusicNetMidiToWavConfig):
    mn_dataset = MusicNetDataset(ds_name, config.mn_ds_type)
    out_dir = os.path.join(DATA_SOURCES_PATH, ds_name)
    track = mn_dataset.get_track(id)
    state = State(ds_name, lock)
    def prepare():
        try:
            midi = track.read_midi_file()
        except BrokenMidiException:
            print("TRACK NOT CONVERTED: BROKEN MIDI FILE", id)
            return False
        
        out_midi_path = os.path.join(out_dir, f"{id}.midi")
        if config.programs_whitelist:
            filtered_midi, p_in, p_out = extract_midi_programs(midi, config.programs_whitelist)
            if len(p_out) == 0:
                print("TRACK NOT CONVERTED: NO OUTPUT PROGRAMS PRESENT", id)
                return False
            filtered_midi.save(out_midi_path)
            print(f"Track {id} midi programs successfully extracted! p_in: {len(p_in)}, p_out: {len(p_out)}")
            return True
        else:
            shutil.copyfile(track.get_midi_path(), out_midi_path)
            return True
    result = prepare()
    state.add_prepared_file(id, result)
    return result
    
def convert_track(id: int, lock, ds_name: str, config: MusicNetMidiToWavConfig):
    mn_dataset = MusicNetDataset(ds_name, config.mn_ds_type)
    track = MusicNetMidiConvertedTrack(id, ds_name, mn_dataset.metadata, config.mn_ds_type)
    state = State(ds_name, lock)
    track.generate_wav()
    print(f"Track {id} successfully converted to wav...")
    state.add_converted_file(id)

def convert(config: MusicNetMidiToWavConfig, ds_name: str):
    with Manager() as manager:
        lock = manager.Lock()
        mn_dataset = MusicNetDataset(ds_name, config.mn_ds_type)
        track_ids = mn_dataset.get_track_ids()

        state = State(ds_name, lock)
        p_stored_ids, p_stored_results = state.get_prepared_ids_and_results()
        c_stored_ids = state.get_converted_ids()

        p_run_ids = [id for id in track_ids if id not in p_stored_ids]

        with Pool(processes=8) as pool:
            p_run_results = pool.starmap(prepare_track, ((id, lock, ds_name, config) for id in p_run_ids))
            p_ids = np.concatenate([np.array(p_stored_ids, dtype=np.int32), np.array(p_run_ids, dtype=np.int32)])
            p_results = np.concatenate([np.array(p_stored_results, dtype=np.bool_), np.array(p_run_results, dtype=np.bool_)])

            c_run_ids = [id for id in np.array(p_ids)[p_results] if id not in c_stored_ids]
            pool.starmap(convert_track, ((id, lock, ds_name, config) for id in c_run_ids))
from musicnet.utils import Track, get_train_ids, get_test_ids, BrokenMidiException
from pathlib import Path
from utils import MidiConvertedTrack, extract_midi_programs, EXTRACTED_MIDI_OUT_DIR, CONVERTED_MIDI_OUT_DIR, load_params
from multiprocessing import Pool
import os
import subprocess
import shutil

params = load_params(["midi_to_wav.*"])

datasets = {
    "train": get_train_ids(),
    # "test": get_test_ids()
}

if os.path.exists(CONVERTED_MIDI_OUT_DIR):
    shutil.rmtree(CONVERTED_MIDI_OUT_DIR)
for ds_name in datasets.keys():
    os.makedirs(os.path.join(CONVERTED_MIDI_OUT_DIR, ds_name), 0o775, exist_ok=True)
if params["programs_whitelist"] and os.path.exists(EXTRACTED_MIDI_OUT_DIR):
    shutil.rmtree(EXTRACTED_MIDI_OUT_DIR)
    os.makedirs(EXTRACTED_MIDI_OUT_DIR, 0o775)

def obtain_input_path(id):
    track = Track(id)
    try:
        midi = track.read_midi_file()
    except BrokenMidiException:
        print("TRACK NOT CONVERTED: BROKEN MIDI FILE", id)
        return None
    if params["programs_whitelist"]:
        filtered_midi, p_in, p_out = extract_midi_programs(midi, params["programs_whitelist"])
        if len(p_out) == 0:
            print("TRACK NOT CONVERTED: NO OUTPUT PROGRAMS PRESENT", id)
            return None
        midi_path = os.path.join(EXTRACTED_MIDI_OUT_DIR, f"{id}.midi")
        filtered_midi.save(midi_path)
        print(f"Track {id} midi programs successfully extracted , p_in: {len(p_in)}, p_out: {len(p_out)}")
        return id, midi_path
    else:
        return id, track.get_midi_path()

def convert_track(id, ds_name, input_path):
    out_file = os.path.join(CONVERTED_MIDI_OUT_DIR, ds_name, f"{id}.wav")
    subprocess.run(
        f"fluidsynth -F {out_file} /usr/share/sounds/sf2/default-GM.sf2 {input_path}",
        shell=True,
        capture_output=True
    )
    if MidiConvertedTrack(id).get_duration() < 1:
        print("TRACK NOT CONVERTED: UNEXPECTED DURATION", id)
        os.remove(out_file)
    else:
        print(f"Track {id} successfully converted to wav...")

with Pool(processes=8) as pool:
    for ds_name, ds_ids in datasets.items():
        input_paths = pool.map(obtain_input_path, ds_ids)
        input_paths = filter(lambda v: v, input_paths)
        pool.starmap(convert_track, [(id, ds_name, input_path) for id, input_path in input_paths])
import os
from ..dataset.base import DATA_SOURCES_PATH
from musicnet.config.dataset.wav_source.SynthMidiToWavConfig import SynthMidiToWavConfig
from musicnet.preprocessing.dataset.SyntheticMidiDataset import SyntheticTrack
from musicnet.utils import recreate_dirs
from .utils import SynthMidiGenerator
from multiprocessing import Pool
from functools import partial

def generate_midi(note: int, ds_name: str, generator: SynthMidiGenerator):
    midi_file = generator.generate_note_track(note)
    midi_path = SyntheticTrack(note, ds_name).get_midi_path()
    midi_file.save(midi_path)
    print(f"Midi track for note {note} successfully generated")

def generate_meta(note: int, ds_name: str, generator: SynthMidiGenerator):
    track = SyntheticTrack(note, ds_name)
    metadata = generator.generate_track_metadata(track)
    metadata.to_csv(track.get_metadata_path())
    print(f"Metadata for note {note} track successfully generated")

def generate_wav(note: int, ds_name: str):
    SyntheticTrack(note, ds_name).generate_wav()
    print(f"Wav track for note {note} successfully generated")

def generate(config: SynthMidiToWavConfig, ds_name: str):
    out_dir = os.path.join(DATA_SOURCES_PATH, ds_name)
    recreate_dirs([out_dir])

    generator = SynthMidiGenerator(config)
    generate_midi_partial = partial(generate_midi, ds_name=ds_name, generator=generator)
    generate_meta_partial = partial(generate_meta, ds_name=ds_name, generator=generator)
    generate_wav_partial = partial(generate_wav, ds_name=ds_name)
    with Pool(8) as pool:
        pool.map(generate_midi_partial, generator.notes)
        pool.map(generate_meta_partial, generator.notes)
        pool.map(generate_wav_partial, generator.notes)
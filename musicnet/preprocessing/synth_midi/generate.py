from musicnet.config.dataset.wav_source.SynthMidiToWavConfig import SynthMidiToWavConfig
from musicnet.preprocessing.dataset.SyntheticMidiDataset import DATASET_MIDIS_PATH, DATASET_WAVS_PATH, SyntheticTrack
from musicnet.utils import recreate_dirs
from .utils import SynthMidiGenerator
from multiprocessing import Pool
from functools import partial

def generate_midi(note, generator):
    generator.generate_note_track(note)
    print(f"Midi track for note {note} successfuly generated")

def generate_wav(note):
    SyntheticTrack(note).generate_wav()
    print(f"Wav track for note {note} successfuly generated")

def generate(config: SynthMidiToWavConfig):
    recreate_dirs([DATASET_MIDIS_PATH, DATASET_WAVS_PATH])

    generator = SynthMidiGenerator(config)
    generate_midi_partial = partial(generate_midi, generator=generator)
    with Pool(8) as pool:
        pool.map(generate_midi_partial, generator.notes)

    with Pool(8) as pool:
        pool.map(generate_wav, generator.notes)
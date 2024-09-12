import os
from ..dataset.base import DATA_SOURCES_PATH
from musicnet.config.dataset.wav_source.SynthMidiToWavConfig import SynthMidiToWavConfig
from musicnet.preprocessing.dataset.SyntheticMidiDataset import SyntheticTrack
from .utils import SynthMidiGenerator
from multiprocessing import Pool, Manager
from enum import Enum

class GenerationStage(Enum):
    MIDI = 0
    META = 1
    WAV = 2

class State():
    def __init__(self, ds_name: str, lock):
        self.state_file_path = os.path.join(DATA_SOURCES_PATH, ds_name, "state")
        self.lock = lock
        if not os.path.exists(self.state_file_path):
            self.set_stage(GenerationStage.MIDI)

    def set_stage(self, stage: GenerationStage):
        with self.lock:
            with open(self.state_file_path, "w") as state_file:
                state_file.write(f"{stage.value}\n")

    def add_processed_note(self, id: int):
        with self.lock:
            with open(self.state_file_path, "a") as state_file:
                state_file.write(f"{id}\n")

    def get_stage_and_processed_notes(self):
        stage = GenerationStage.MIDI
        processed_notes = []
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, "r") as state_file:
                stage = GenerationStage(int(state_file.readline()))
                for l in state_file.readlines():
                    if l:
                        processed_notes.append(int(l))
        return stage, processed_notes

def generate_midi(note: int, lock, ds_name: str, config: SynthMidiToWavConfig):
    state = State(ds_name, lock)
    generator = SynthMidiGenerator(config)
    midi_file = generator.generate_note_track(note)
    midi_path = SyntheticTrack(note, ds_name).get_midi_path()
    midi_file.save(midi_path)
    print(f"Midi track for note {note} successfully generated")
    state.add_processed_note(note)

def generate_meta(note: int, lock, ds_name: str, config: SynthMidiToWavConfig):
    state = State(ds_name, lock)
    generator = SynthMidiGenerator(config)
    track = SyntheticTrack(note, ds_name)
    metadata = generator.generate_track_metadata(track)
    metadata.to_csv(track.get_metadata_path())
    print(f"Metadata for note {note} track successfully generated")
    state.add_processed_note(note)

def generate_wav(note: int, lock, ds_name: str):
    state = State(ds_name, lock)
    SyntheticTrack(note, ds_name).generate_wav()
    print(f"Wav track for note {note} successfully generated")
    state.add_processed_note(note)

def generate(config: SynthMidiToWavConfig, ds_name: str):
    with Manager() as manager:
        lock = manager.Lock()
        generator = SynthMidiGenerator(config)
        state = State(ds_name, lock)
        stored_stage, stored_p_notes = state.get_stage_and_processed_notes()
        remaining_notes = set(generator.notes) - set(stored_p_notes)
        print(f"Initial state: stage={stored_stage.name}, remaining_notes={len(remaining_notes)}")
        with Pool(8) as pool:
            if stored_stage.value <= GenerationStage.MIDI.value:
                pool.starmap(generate_midi, ((note, lock, ds_name, config) for note in remaining_notes))
                remaining_notes = generator.notes
                state.set_stage(GenerationStage.META)
            if stored_stage.value <= GenerationStage.META.value:
                pool.starmap(generate_meta, ((note, lock, ds_name, config) for note in remaining_notes))
                remaining_notes = generator.notes
                state.set_stage(GenerationStage.WAV)
            if stored_stage.value <= GenerationStage.WAV.value:
                pool.starmap(generate_wav, ((note, lock, ds_name) for note in remaining_notes))
from musicnet.config.dataset.wav_source.SynthMidiToWavConfig import SynthMidiToWavConfig
from musicnet.preprocessing.dataset.SyntheticMidiDataset import DATASET_MIDIS_PATH
from mido import MetaMessage, Message, MidiFile, MidiTrack, bpm2tempo
import os
import pandas as pd
import random
import numpy as np
import scipy

class AccompanyingNotesGenerator:
    def __init__(self, main_note, notes, notes_std, batch_size=10000) -> None:
        self.batch_size = batch_size
        self.batch = []
        self.notes = notes
        p = np.array([scipy.stats.norm.pdf(n, loc=main_note, scale=notes_std) if n != main_note else 0 for n in notes])
        p = p / p.sum()
        self.p = p
    
    def next(self):
        if len(self.batch) == 0:
            self.batch = list(np.random.choice(self.notes, size=self.batch_size, p=self.p))
        return self.batch.pop()
    
class SynthMidiGenerator:
    def __init__(self, config: SynthMidiToWavConfig):
        self.tps = config.tps
        self.min_note_ticks = config.min_note_ticks
        self.max_note_ticks = config.max_note_ticks
        self.velocity_min = config.velocity_min
        self.velocity_max = config.velocity_max
        self.max_silmultaneous_notes = config.max_silmultaneous_notes
        self.notes = list(range(config.min_note, config.max_note))
        self.notes_std = config.notes_std
        self.sidechannel_sparsity = config.sidechannel_sparsity
        self.total_ticks = self.tps * config.track_length_per_note

    def generate_channel_events(
        self,
        main_note: int,
        is_sidechannel=False,
        channel=0,
        program=0
    ):
        tick = 0
        events = [{ "tick": 0, "type": "program_change", "channel": channel, "program": program }]
        generator = AccompanyingNotesGenerator(main_note, self.notes, self.notes_std)
        current_note = None
        while tick < self.total_ticks:
            event: dict[str, int | str] = {
                "tick": tick,
                "channel": channel
            }
            if current_note is None:
                event["type"] = "note_on"
                event["velocity"] = random.randint(self.velocity_min, self.velocity_max)
                current_note = generator.next() if is_sidechannel else main_note
                event["note"] = current_note
            else:
                event["type"] = "note_off"
                event["note"] = current_note
                current_note = None
            events.append(event)
            tick_delta = random.randint(self.min_note_ticks, self.max_note_ticks)
            if current_note is None:
                tick_delta *= self.sidechannel_sparsity if is_sidechannel else 1
            tick += tick_delta
        return events

    def generate_note_track(self, note):
        events = []
        # Main channel events
        events = events + self.generate_channel_events(main_note=note, is_sidechannel=False, channel=0)
        # "Side" channels events
        for c in range(1, self.max_silmultaneous_notes):
            events = events + self.generate_channel_events(
                main_note=note,
                is_sidechannel=True,
                channel=c
            )
        events = pd.DataFrame(events).sort_values(by=["tick", "type"], ascending=[True, False]).reset_index(drop=True)
        # Calculate time as delta in ticks
        events["time"] = (events["tick"] - events["tick"].shift(1).fillna(0)).astype(int)
        events["velocity"] = events["velocity"].astype(pd.Int8Dtype())
        events["program"] = events["program"].astype(pd.Int8Dtype())
        events["note"] = events["note"].astype(pd.Int8Dtype())
        events = events.drop(columns=["tick"])
        # Convert events to midi messages
        note_messages = [ Message(**{ k: v for k, v in event.items() if pd.notnull(v) and type(k) == str }) for event in events.to_dict("records")]
        # Create the output midi file
        file = MidiFile(ticks_per_beat=self.tps)
        # Set tempo to 60 bpm, so 1 beat / second to simplify calculations
        messages = [MetaMessage("set_tempo", tempo=bpm2tempo(60), time=0)]
        messages += note_messages
        track = MidiTrack(messages)
        file.tracks.append(track)
        midi_path = os.path.join(DATASET_MIDIS_PATH, f"{note}.midi")
        file.save(midi_path)

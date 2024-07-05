import os
from glob import glob
import pandas as pd
import mido
import numpy as np
import librosa
import math

DATASET_BASE_PATH = "/home/leszek/datasets/MusicNet"
FILES_BASE_PATH = DATASET_BASE_PATH + "/musicnet/musicnet"

def create_vocab(value_set):
    values = list(value_set)
    values.sort()
    return { value: index for index, value in enumerate(values) }

def load_instruments_and_notes():
    instruments = set()
    notes = set()
    for file in glob(os.path.join(FILES_BASE_PATH, "train_labels/*.csv")):
        data = pd.read_csv(file)
        for i in data["instrument"].unique():
            instruments.add(i)
        for n in data["note"].unique():
            notes.add(n)

    return create_vocab(instruments), create_vocab(notes)

metadata = pd.read_csv(os.path.join(DATASET_BASE_PATH, "musicnet_metadata.csv")).set_index("id")
instruments_vocab, notes_vocab = load_instruments_and_notes()

class Track:
    def __init__(self, _id):
        self._id = _id
        if os.path.exists(os.path.join(FILES_BASE_PATH, "train_data", f"{self._id}.wav")):
            self.ds = "train"
        elif os.path.exists((os.path.join(FILES_BASE_PATH, "test_data", f"{self._id}.wav"))):
            self.ds = "test"
        else:
            raise Exception(f"Track {_id} not found!")
        self.metadata = metadata.loc[_id]
    
    def get_wav_path(self):
        return os.path.join(FILES_BASE_PATH, f"{self.ds}_data", f"{self._id}.wav")
    
    def get_csv_path(self):
        return os.path.join(FILES_BASE_PATH, f"{self.ds}_labels", f"{self._id}.csv")
    
    def get_metadata(self):
        return self.metadata
    
    def get_midi_path(self):
        pattern = os.path.join(
            DATASET_BASE_PATH,
            "musicnet_midis",
            "musicnet_midis",
            self.metadata["composer"],
            f"{self._id}_*.mid"
        )
        found = glob(pattern)
        if len(found):
            return found[0]
        else:
            raise Exception(f"Track {self._id} midi file not found!")
        
    def read_csv_data(self):
        return pd.read_csv(self.get_csv_path())
    
    def read_wav_data(self, sr=None):
        return librosa.get_samplerate(self.get_wav_path()), librosa.load(self.get_wav_path(), sr=sr)[0]
    
    def read_midi_notes(self):
        mid = mido.MidiFile(self.get_midi_path())
        notes = []
        note_start_times = np.zeros(shape=(16, 128))
        curr_time = 0
        for msg in mid:
            if msg.is_meta:
                continue
            curr_time += msg.time
            if msg.type == "note_on":
                note_start_times[msg.channel][msg.note] = curr_time
            if msg.type == "note_off":
                if note_start_times[msg.channel][msg.note] == 0:
                    raise Exception(f"Empty note found!")
                notes.append({
                    "note": msg.note,
                    "channel": msg.channel,
                    "start_time": note_start_times[msg.channel][msg.note],
                    "end_time": curr_time
                })
                note_start_times[msg.channel][msg.note] = 0
        return pd.DataFrame(notes)
    
    def count_chunks(
        self,
        chunk_size_sec=10,
        chunk_shift_sec=5
    ):
        duration_sec = librosa.get_duration(path=self.get_wav_path())
        num_chunks = 1 + max(0, math.ceil((duration_sec - chunk_size_sec) / chunk_shift_sec))
        return num_chunks
    
    def preprocess(
        self,
        chunk_size_sec=10,
        chunk_shift_sec=5,
        target_sr=16000,
        note_rounding=0.01,
        # Number of samples in a window per fft in spectrogram (25 ms)
        n_fft = 400,
        # The amount of samples we are shifting after each fft in spectrogram (10 ms)
        hop_length = 160,
        # Number of filters in mel spectrogram
        n_mels=128,
    ):
        csv_data = self.read_csv_data()
        src_sr, wav_data = self.read_wav_data(target_sr)
        csv_data["start_time_s"] = np.round(csv_data["start_time"] / src_sr / note_rounding) * note_rounding
        csv_data["end_time_s"] = np.round(csv_data["end_time"] / src_sr / note_rounding) * note_rounding
        
        num_chunks = self.count_chunks(chunk_shift_sec, chunk_shift_sec)
        padded_track_end_sec = chunk_size_sec + (num_chunks - 1) * chunk_shift_sec
        wav_data = np.pad(wav_data, (0, padded_track_end_sec * target_sr - len(wav_data)))

        x_chunks = []
        y_chunks = []

        for c in range(0, num_chunks):
            start_s = c * chunk_shift_sec
            end_s = start_s + chunk_size_sec
            wav_chunk = wav_data[start_s * target_sr : end_s * target_sr]
            chunk_notes = csv_data[(csv_data["start_time_s"] >= start_s) & (csv_data["start_time_s"] < end_s - note_rounding)].copy()
            
            seq = np.zeros(shape=(int(chunk_size_sec / note_rounding), len(notes_vocab) * len(instruments_vocab)))
            for note in chunk_notes.to_dict("records"):
                start_idx = int(np.round((note["start_time_s"] - start_s) / note_rounding))
                end_idx = min(seq.shape[0], int(np.round((note["end_time_s"] - start_s) / note_rounding)))
                note_idx = notes_vocab[note["note"]] + instruments_vocab[note["instrument"]] * len(notes_vocab)
                seq[start_idx : end_idx, note_idx] = 1
            y_chunks.append(seq)

            mel_spec = librosa.feature.melspectrogram(
                y=wav_chunk,
                sr=target_sr,
                hop_length=hop_length,
                n_fft=n_fft,
                n_mels=n_mels,
                center=False
            )
            mel_spec_db = librosa.power_to_db(mel_spec)
            x_chunks.append(mel_spec_db.T)
        return np.array(x_chunks), np.array(y_chunks)

def list_track_ids(ds_type="train"):
    return list(map(
        lambda f: int(f.split("/")[-1].split('.')[0]),
        glob(os.path.join(FILES_BASE_PATH, f"{ds_type}_labels/*.csv"))
    ))

train_ids = lambda : list_track_ids("train")
test_ids = lambda : list_track_ids("test")
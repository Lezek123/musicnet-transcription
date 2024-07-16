import librosa
import math
import numpy as np
import tensorflow as tf
import os
from glob import glob
from dataclasses import dataclass
from musicnet.utils import Track, notes_vocab, PROJECT_ROOT_DIR
from typing import Literal
from numpy.fft import rfftfreq
import random


PREPROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "preprocessed", "wav_nmel_to_notes")

def get_out_dir(from_midis = False):
    if from_midis:
        return os.path.join(PREPROCESSED_DATA_DIR, "from_midis")
    else:
        return os.path.join(PREPROCESSED_DATA_DIR, "from_wavs")

@dataclass
class Preprocessor():
    chunk_size_sec: int
    chunk_shift_sec: int
    target_sr: int
    note_rounding: float
    # Number of samples in a window per fft in spectrogram
    n_fft: int
    # The amount of samples we are shifting after each fft in spectrogram
    hop_length: int
    # Minimum spectogram frequency in Hz
    min_hz: int
    # Number of filters in the spectrogram (also determines max_hz)
    n_filters: int
    # Instruments vocabulary to use (missing instruments will be omitted)
    instruments_vocab: dict[int, int]
    unit: Literal["amplitude", "decibels"]
        
    def count_chunks(self, track: Track):
        duration_sec = track.get_duration()
        num_chunks = 1 + max(0, math.ceil((duration_sec - self.chunk_size_sec) / self.chunk_shift_sec))
        return num_chunks
    
    def round(self, column):
        return np.round(column / self.note_rounding) * self.note_rounding
    
    def create_spectogram(self, signal):
        audio_stft = librosa.core.stft(
            signal,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        amp_spectrogram = np.abs(audio_stft)
        hz_frequencies = rfftfreq(self.n_fft, 1.0 / self.target_sr)
        start_index = np.argmax(hz_frequencies >= self.min_hz)
        end_index = start_index + self.n_filters
        amp_spectrogram = amp_spectrogram[start_index:end_index, :]
        if self.unit == "amplitude":
            return amp_spectrogram
        else:
            return librosa.amplitude_to_db(amp_spectrogram)

    def preprocess(self, track: Track):
        instruments_vocab = self.instruments_vocab
        wav_data = track.read_wav_data(self.target_sr)
        notes = track.get_notes()
        notes["start"] = self.round(notes["start"])
        notes["end"] = self.round(notes["end"])
        
        num_chunks = self.count_chunks(track)
        padded_track_end_sec = self.chunk_size_sec + (num_chunks - 1) * self.chunk_shift_sec
        wav_data = np.pad(wav_data, (0, padded_track_end_sec * self.target_sr - len(wav_data)))

        x_chunks = []
        y_chunks = []

        for c in range(0, num_chunks):
            start_s = c * self.chunk_shift_sec
            end_s = start_s + self.chunk_size_sec
            wav_chunk = wav_data[start_s * self.target_sr : end_s * self.target_sr]
            chunk_notes = notes[(notes["start"] >= start_s) & (notes["start"] < end_s - self.note_rounding)].copy()
            
            seq = np.zeros(shape=(int(self.chunk_size_sec / self.note_rounding), len(notes_vocab) * len(instruments_vocab)))
            for note in chunk_notes.to_dict("records"):
                # Ignore instruments not present in the vocab
                if not (note["instrument"] in instruments_vocab):
                    continue
                start_idx = int(np.round((note["start"] - start_s) / self.note_rounding))
                end_idx = min(seq.shape[0], int(np.round((note["end"] - start_s) / self.note_rounding)))
                note_idx = notes_vocab[note["note"]] + instruments_vocab[note["instrument"]] * len(notes_vocab)
                seq[start_idx : end_idx, note_idx] = 1
            y_chunks.append(seq)

            spec = self.create_spectogram(wav_chunk)
            x_chunks.append(spec.T)
        return np.array(x_chunks), np.array(y_chunks)
    
def decode_record(record_bytes, n_filters, target_classes, architecture="encoder-decoder"):
    example = tf.io.parse_example(record_bytes, {
        "x": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "y": tf.io.FixedLenFeature([], tf.string, default_value="")
    })
    x = tf.io.parse_tensor(example["x"], tf.float32)
    x.set_shape([None, n_filters])
    y = tf.io.parse_tensor(example["y"], tf.bool)
    y.set_shape([None, target_classes])

    if architecture == "encoder-decoder":
        return (x, y[:-1]), y[1:]
    else:
        return (x, y)

def create_tf_record_ds(
    ds_type,
    n_filters,
    target_classes,
    batch_size,
    architecture,
    use_converted_midis=False,
    dataset_size=1.0,
    num_parallel_reads="auto"
):
    source_dir = os.path.join(get_out_dir(use_converted_midis), ds_type)
    files = glob(os.path.join(source_dir, "*.tfrecord"))
    random.shuffle(files)
    files = files[:int(dataset_size * len(files))]

    if num_parallel_reads == 'auto':
        num_parallel_reads = len(files)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    ds = ds.map(lambda r: decode_record(r, n_filters, target_classes, architecture)).shuffle(1000).batch(batch_size).prefetch(1)
    return ds
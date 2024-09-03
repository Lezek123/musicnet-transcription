import librosa
import math
import numpy as np
import tensorflow as tf
import os
from glob import glob
from musicnet.utils import IS_CLOUD
from numpy.fft import rfftfreq
from musicnet.config.dataset.preprocessor.WavChunksTFRecordPreprocessorConfig import (
    WavChunksTFRecordPreprocessorParams,
    WavChunksTFRecordPreprocessorConfig
)
from musicnet.config.dataset.DatasetConfig import DsConfig
from ..dataset.base import BaseTrack, PREPROCESSED_DATA_PATH
import psutil

class Preprocessor():
    def __init__(self, config: WavChunksTFRecordPreprocessorParams, notes_vocab: dict[int, int], instruments_vocab: dict[int, int]):
        self.chunk_size_sec = config.chunk_size_sec
        self.chunk_shift_sec = config.chunk_shift_sec
        self.target_sr = config.target_sr
        self.note_rounding = config.note_rounding
        self.spectogram = config.spectogram
        self.notes_vocab = notes_vocab
        self.instruments_vocab = instruments_vocab

        
    def count_chunks(self, track: BaseTrack):
        duration_sec = track.get_duration()
        num_chunks = 1 + max(0, math.ceil((duration_sec - self.chunk_size_sec) / self.chunk_shift_sec))
        return num_chunks
    
    def round(self, column):
        return np.round(column / self.note_rounding) * self.note_rounding
    
    def create_spectogram(self, signal):
        params = self.spectogram
        if not params:
            raise Exception("Cannot create spectogram: Missing parameters")
        audio_stft = librosa.core.stft(
            signal,
            hop_length=params.hop_length,
            n_fft=params.n_fft
        )
        amp_spectrogram = np.abs(audio_stft)
        hz_frequencies = rfftfreq(params.n_fft, 1.0 / self.target_sr)
        start_index = np.argmax(hz_frequencies >= params.min_hz)
        end_index = start_index + params.n_filters
        amp_spectrogram = amp_spectrogram[start_index:end_index, :]
        if params.unit == "amplitude":
            return amp_spectrogram
        else:
            return librosa.amplitude_to_db(amp_spectrogram)

    def preprocess(self, track: BaseTrack):
        instruments_vocab = self.instruments_vocab
        notes_vocab = self.notes_vocab
        wav_data = track.read_wav_data(self.target_sr)
        notes = track.get_notes()
        notes["start"] = self.round(notes["start"])
        notes["end"] = self.round(notes["end"])
        
        num_chunks = self.count_chunks(track)
        padded_track_end_sec = self.chunk_size_sec + (num_chunks - 1) * self.chunk_shift_sec
        wav_data = np.pad(wav_data, (0, int(padded_track_end_sec * self.target_sr) - len(wav_data)))

        x_chunks = []
        y_chunks = []

        for c in range(0, num_chunks):
            start_s = c * self.chunk_shift_sec
            end_s = start_s + self.chunk_size_sec
            wav_chunk = wav_data[int(start_s * self.target_sr) : int(end_s * self.target_sr)].copy()
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

            if self.spectogram:
                spec = self.create_spectogram(wav_chunk)
                x_chunks.append(spec.T)
            else:
                x_chunks.append(wav_chunk.reshape(-1, 1))
        return np.array(x_chunks), np.array(y_chunks)
    
def decode_record(record_bytes, x_shape, y_shape):
    example = tf.io.parse_example(record_bytes, {
        "x": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "y": tf.io.FixedLenFeature([], tf.string, default_value=""),
    })
    x: tf.Tensor = tf.io.parse_tensor(example["x"], tf.float32)
    y: tf.Tensor = tf.io.parse_tensor(example["y"], tf.float32)
    x.set_shape(x_shape)
    y.set_shape(y_shape)

    return (x, y)

def load_shapes(ds_name: str):
    x_shape, y_shape = np.load(os.path.join(PREPROCESSED_DATA_PATH, ds_name, "shapes.npy"))
    return x_shape, y_shape

def save_shapes(x_shape, y_shape, ds_name: str):
    return np.save(os.path.join(PREPROCESSED_DATA_PATH, ds_name, "shapes.npy"), [x_shape, y_shape])

def create_tf_record_ds(
    ds_config: DsConfig,
    ds_name: str,
    num_parallel_reads="auto",
    buffer_size=None,
    shuffle=True
):
    if not isinstance(ds_config.preprocessor, WavChunksTFRecordPreprocessorConfig):
        raise Exception("Unsupported preprocessor")

    ds_dir = os.path.join(PREPROCESSED_DATA_PATH, ds_name)
    files = glob(os.path.join(ds_dir, "*.tfrecord"))
    files = sorted(files)

    if num_parallel_reads == 'auto':
        num_parallel_reads = len(files)
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    
    x_shape, y_shape = load_shapes(ds_name)
    ds = ds.map(lambda r: decode_record(r, x_shape, y_shape))

    buffer_size = buffer_size or 0.25

    if type(buffer_size) == float:
        record_size = (np.prod(x_shape) + np.prod(y_shape)) * 32
        available_mem = psutil.virtual_memory().available
        buffer_size = int((buffer_size * available_mem) / record_size)

    if os.environ.get("DS_CACHE") == "true":
        print("DS_CACHE enabled")
        ds = ds.cache()
        buffer_size = ds.cardinality()

    print(f"{ds_name} dataset buffer size: {buffer_size}")

    if shuffle:
        ds = ds.shuffle(buffer_size)

    return ds.batch(ds_config.batch_size).prefetch(1)
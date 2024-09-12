from .utils import Preprocessor, save_shapes, shapes_file_path
from musicnet.config.dataset.preprocessor import WavChunksTFRecordPreprocessorConfig
from musicnet.utils import recreate_dirs
from ..dataset.base import BaseDataset, PREPROCESSED_DATA_PATH
from typing import Tuple, Optional
from multiprocessing import Pool
import tensorflow as tf
import os
import numpy as np
import psutil

class State():
    def __init__(self, ds_name: str):
        self.state_file_path = os.path.join(PREPROCESSED_DATA_PATH, ds_name, "state")

    def set_num_prepared_files(self, num: int):
        with open(self.state_file_path, "w") as state_file:
            state_file.write(str(num))
    
    def load_num_prepared_files(self):
        if not os.path.exists(self.state_file_path):
            return 0
        
        with open(self.state_file_path, "r") as state_file:
            return int(state_file.readline())

def preprocess_track(id: int, dataset: BaseDataset, preprocessor: Preprocessor):
    track = dataset.get_track(id)
    x_chunks, y_chunks = preprocessor.preprocess(track)
    return x_chunks, y_chunks

class ChunkIterator:
    def __init__(
        self,
        dataset: BaseDataset,
        preprocessor: Preprocessor,
        track_ids: list[int],
        start_chunk: Optional[int] = None,
        preload_tracks_n = 4
    ):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.track_ids = track_ids.copy()
        self.start_chunk = start_chunk
        self.preload_tracks_n = preload_tracks_n

    def move_to_start_track(self):
        if not self.start_chunk:
            return
        for i in range(0, len(self.track_ids)):
            track = self.dataset.get_track(self.track_ids[i])
            chunks_count = self.preprocessor.count_chunks(track)
            if self.current_chunk + chunks_count < self.start_chunk:
                self.current_chunk += chunks_count
                print(f"skipped track {self.track_ids[i]}")
            else:
                self.current_track = i
                break

    def __iter__(self):
        self.x_chunks: list[np.ndarray] = []
        self.y_chunks: list[np.ndarray] = []
        self.current_track = 0
        self.current_chunk = 0
        self.move_to_start_track()
        return self

    def __next__(self):
        if (len(self.x_chunks) == 0):
            preload_ids = self.track_ids[self.current_track : self.current_track + self.preload_tracks_n]
            print(f"Memory usage pre-load: {psutil.Process().memory_info().rss / 1024 / 1024} MiB")
            with Pool(self.preload_tracks_n) as pool:
                preloaded_chunks = pool.starmap(preprocess_track, ((id, self.dataset, self.preprocessor) for id in preload_ids))
            print(f"loaded tracks {preload_ids}")
            print(f"Memory usage post-load: {psutil.Process().memory_info().rss / 1024 / 1024} MiB")
            self.current_track += self.preload_tracks_n
            self.x_chunks = list(np.concatenate([x_chunks_batch for x_chunks_batch, _ in preloaded_chunks]))
            self.y_chunks = list(np.concatenate([y_chunks_batch for _, y_chunks_batch in preloaded_chunks]))
            print(f"Memory usage post-assign: {psutil.Process().memory_info().rss / 1024 / 1024} MiB")
            if self.start_chunk and (self.current_chunk < self.start_chunk):
                remaining_to_skip = self.start_chunk - self.current_chunk
                self.x_chunks = self.x_chunks[remaining_to_skip:]
                self.y_chunks = self.y_chunks[remaining_to_skip:]
                self.current_chunk = self.start_chunk
                print(f"skipped {remaining_to_skip} chunks from track {preload_ids[0]}")
            self.current_chunk += 1
        return self.x_chunks.pop(0), self.y_chunks.pop(0)
    
def serialize_chunks(x_chunk, y_chunk):
    x_chunk = tf.constant(x_chunk, dtype=tf.float32)
    y_chunk = tf.constant(y_chunk, dtype=tf.float32)
    x = tf.io.serialize_tensor(x_chunk)
    y = tf.io.serialize_tensor(y_chunk)
    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.numpy()])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.numpy()])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def preprocess(
    config: WavChunksTFRecordPreprocessorConfig,
    dataset: BaseDataset,
    ds_name: str,
    instruments_vocab: dict[int, int],
    notes_vocab: dict[int, int],
    split: Optional[Tuple[float, float]]
):
    out_dir = os.path.join(PREPROCESSED_DATA_PATH, ds_name)

    track_ids = sorted(dataset.get_track_ids())
    state = State(ds_name)

    preprocessor = Preprocessor(
        config.params,
        notes_vocab=notes_vocab,
        instruments_vocab=instruments_vocab
    )

    num_prepared_files = state.load_num_prepared_files()

    if num_prepared_files == config.file_count:
        print("All files already preprocessed, skipping...")
        return
    
    print(f"processing {ds_name} set".upper())

    total_chunks = 0
    start_chunk = 0
    
    for id in track_ids:
        track = dataset.get_track(id)
        total_chunks += preprocessor.count_chunks(track)

    print(f"Total source chunks: {total_chunks}")
    print(f"Total output files: {config.file_count}")

    if split:
        start_chunk = int(total_chunks * split[0])
        end_chunk = int(total_chunks * split[1])
        total_chunks = end_chunk - start_chunk
        print(f"Total [{split[0]} - {split[1]}] split chunks: {total_chunks} ({start_chunk} - {end_chunk})")

    chunks_per_file = int(total_chunks / config.file_count)

    if num_prepared_files > 0:
        start_chunk = start_chunk + chunks_per_file * num_prepared_files

    print(f"Chunks per file: {chunks_per_file}")
    print(f"Already prepared files: {num_prepared_files}")    
    print(f"Start chunk: {start_chunk}")

    chunks = ChunkIterator(dataset, preprocessor, track_ids, start_chunk)
    chunks = iter(chunks)

    shapes_saved = os.path.exists(shapes_file_path(ds_name))

    for i in range(num_prepared_files, config.file_count):
        filepath = os.path.join(out_dir, f"{str(i+1).zfill(3)}.tfrecord")
        x_shape, y_shape = None, None
        with tf.io.TFRecordWriter(filepath) as writer:
            for _ in range(0, chunks_per_file):
                x_chunk, y_chunk = next(chunks)
                x_shape, y_shape = x_chunk.shape, y_chunk.shape
                serialized = serialize_chunks(x_chunk, y_chunk)
                writer.write(serialized)
        print(f"created {filepath}")
        if not shapes_saved and (x_shape and y_shape):
            save_shapes(x_shape, y_shape, ds_name)
            shapes_saved = True
        state.set_num_prepared_files(i+1)

    print("Done!")
    print("\n\n")
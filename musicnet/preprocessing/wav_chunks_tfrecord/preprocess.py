from .utils import Preprocessor, save_shapes
from musicnet.config.dataset.preprocessor import WavChunksTFRecordPreprocessorConfig
from musicnet.utils import recreate_dirs
from ..dataset.base import BaseDataset, PREPROCESSED_DATA_PATH
from typing import Tuple, Optional
import tensorflow as tf
import os
import numpy as np

def preprocess(
    config: WavChunksTFRecordPreprocessorConfig,
    dataset: BaseDataset,
    ds_name: str,
    instruments_vocab: dict[int, int],
    notes_vocab: dict[int, int],
    split: Optional[Tuple[float, float]]
):
    out_dir = os.path.join(PREPROCESSED_DATA_PATH, ds_name)
    recreate_dirs([out_dir])

    track_ids = sorted(dataset.get_track_ids())

    preprocessor = Preprocessor(
        config.params,
        notes_vocab=notes_vocab,
        instruments_vocab=instruments_vocab
    )

    class ChunkIterator:
        def __init__(self, track_ids: list[int], start_chunk: Optional[int] = None):
            self.track_ids = track_ids.copy()
            self.start_chunk = start_chunk
        
        def move_to_start_track(self):
            if not self.start_chunk:
                return
            for i in range(0, len(self.track_ids)):
                track = dataset.get_track(self.track_ids[i])
                chunks_count = preprocessor.count_chunks(track)
                if self.current_chunk + chunks_count < start_chunk:
                    self.current_chunk += chunks_count
                    print(f"skipped track {id}")
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
                id = track_ids.pop()
                track = dataset.get_track(id)
                print(f"loaded track {id}")
                x_chunks, y_chunks = preprocessor.preprocess(track)

                self.x_chunks += list(x_chunks)
                self.y_chunks += list(y_chunks)

                if self.start_chunk and (self.current_chunk < self.start_chunk):
                    remaining_to_skip = self.start_chunk - self.current_chunk
                    self.x_chunks = self.x_chunks[remaining_to_skip:]
                    self.y_chunks = self.y_chunks[remaining_to_skip:]
                    self.current_chunk = self.start_chunk
                    print(f"skipped {remaining_to_skip} chunks from track {id}")

                self.current_chunk += 1
            return self.x_chunks.pop(0), self.y_chunks.pop(0)
        
    total_chunks = 0
    for id in track_ids:
        track = dataset.get_track(id)
        total_chunks += preprocessor.count_chunks(track)

    if split:
        start_chunk = int(total_chunks * split[0])
        end_chunk = int(total_chunks * split[1])
        total_chunks = end_chunk - start_chunk
        chunks = ChunkIterator(track_ids, start_chunk)
    else:
        chunks = ChunkIterator(track_ids)

    print(f"processing {ds_name} set".upper())
    print(f"Total chunks: {total_chunks}")

    def serialize(x_chunk, y_chunk):
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

    ds_files = config.file_count
    chunks_per_file = int(total_chunks / ds_files)
    chunks = iter(chunks)
    
    print(f"num files: {ds_files}, chunks per file: {chunks_per_file}")
    for i in range(0, ds_files):
        filepath = os.path.join(out_dir, f"{str(i).zfill(3)}.tfrecord")
        x_shape, y_shape = None, None
        with tf.io.TFRecordWriter(filepath) as writer:
            for _ in range(0, chunks_per_file):
                x_chunk, y_chunk = next(chunks)
                x_shape, y_shape = x_chunk.shape, y_chunk.shape
                serialized = serialize(x_chunk, y_chunk)
                writer.write(serialized)
        print(f"created {filepath}")
    if x_shape and y_shape:
        save_shapes(x_shape, y_shape, ds_name)
    print("Done!")
    print("\n\n")
from .utils import Preprocessor, PREPROCESSED_DATA_DIR, save_shapes
from musicnet.config.dataset.preprocessor import WavChunksTFRecordPreprocessorConfig
from musicnet.config.dataset.wav_source import WavSourceType
from ..dataset.base import BaseDataset
import random
import tensorflow as tf
import os
import shutil
import numpy as np

def preprocess(config: WavChunksTFRecordPreprocessorConfig, dataset: BaseDataset, wav_source_type: WavSourceType):
    out_dir = os.path.join(PREPROCESSED_DATA_DIR, wav_source_type.value)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, 0o775, exist_ok=True)

    track_ids = dataset.get_track_ids()
    random.shuffle(track_ids)

    preprocessor = Preprocessor(
        config.params,
        notes_vocab=dataset.notes_vocab,
        instruments_vocab=dataset.instruments_vocab
    )

    class ChunkIterator:
        def __iter__(self):
            self.x_chunks: list[np.ndarray] = []
            self.y_chunks: list[np.ndarray] = []
            return self

        def __next__(self):
            if (len(self.x_chunks) == 0):
                id = track_ids.pop()
                track = dataset.get_track(id)
                print(f"loaded track {id}")
                x_chunks, y_chunks = preprocessor.preprocess(track)

                self.x_chunks += list(x_chunks)
                self.y_chunks += list(y_chunks)
            return self.x_chunks.pop(0), self.y_chunks.pop(0)
        
    total_chunks = 0
    for id in track_ids:
        track = dataset.get_track(id)
        total_chunks += preprocessor.count_chunks(track)

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

    chunks_iterator = ChunkIterator()
    # FIXME:
    # We could also do it inside the loop to prevent
    # continuation of the train track from entering val set,
    # but then the chunks count will get messed up
    chunks = iter(chunks_iterator)

    for ds_type, ds_config in config.ds_split.items():
        print(f"processing {ds_type} set".upper())
        ds_size = ds_config.size
        ds_files = ds_config.file_count
        if ds_size > 1:
            chunks_per_file = int(ds_size / ds_files)
        else:
            chunks_per_file = int((ds_size * total_chunks) / ds_files)
        print(f"num files: {ds_files}, chunks per file: {chunks_per_file}")
        ds_dir = os.path.join(out_dir, ds_type)
        os.makedirs(ds_dir, 0o775, exist_ok=True)
        for i in range(0, ds_files):
            filepath = os.path.join(ds_dir, f"{str(i).zfill(3)}.tfrecord")
            x_shape, y_shape = None, None
            with tf.io.TFRecordWriter(filepath) as writer:
                for _ in range(0, chunks_per_file):
                    x_chunk, y_chunk = next(chunks)
                    x_shape, y_shape = x_chunk.shape, y_chunk.shape
                    serialized = serialize(x_chunk, y_chunk)
                    writer.write(serialized)
            print(f"created {filepath}")
        if x_shape and y_shape:
            save_shapes(x_shape, y_shape, wav_source_type)
        print("Done!")
        print("\n\n")
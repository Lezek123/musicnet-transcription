from musicnet.utils import Track, train_ids, load_params
import random
import tensorflow as tf
import os
from pathlib import Path

params = load_params(("preprocess", "shared"))

ids_train_val = train_ids()
random.shuffle(ids_train_val)

class ChunkIterator:
    def __iter__(self):
        self.x_chunks = []
        self.y_chunks = []
        return self

    def __next__(self):
        if (len(self.x_chunks) == 0):
            id = ids_train_val.pop()
            track = Track(id)
            print(f"loaded track {id}")
            x_chunks, y_chunks = track.preprocess()
            self.x_chunks += list(x_chunks)
            self.y_chunks += list(y_chunks)
        return self.x_chunks.pop(0), self.y_chunks.pop(0)
    
total_chunks = 0
for id in ids_train_val:
    total_chunks += Track(id).count_chunks()

print(f"Total chunks: {total_chunks}")

def serialize(x_chunk, y_chunk):
    x_chunk = tf.constant(x_chunk, dtype=tf.float32)
    y_chunk = tf.constant(y_chunk, dtype=tf.bool)
    x = tf.io.serialize_tensor(x_chunk)
    y = tf.io.serialize_tensor(y_chunk)
    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.numpy()])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

chunks = ChunkIterator()

for ds_type, ds_config in params["datasets"].items():
    print(f"processing {ds_type} set".upper())
    chunks_iter = iter(chunks)
    ds_size = ds_config["size"]
    ds_files = ds_config["file_count"]
    if ds_size > 1:
        chunks_per_file = int(ds_size / ds_files)
    else:
        chunks_per_file = int((ds_size * total_chunks) / ds_files)
    print(f"num files: {ds_files}, chunks per file: {chunks_per_file}")
    ds_dir = str(Path(__file__).parent.with_name("data").joinpath(ds_type))
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir, 0o775)
    for i in range(0, ds_config["file_count"]):
        filepath = os.path.join(ds_dir, f"{str(i).zfill(3)}.tfrecord")
        with tf.io.TFRecordWriter(filepath) as writer:
            for j in range(0, chunks_per_file):
                x_chunk, y_chunk = next(chunks)
                serialized = serialize(x_chunk, y_chunk)
                writer.write(serialized)
        print(f"created {filepath}")
    print("Done!")
    print("\n\n")
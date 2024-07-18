from musicnet.utils import Track, get_train_ids, instruments_vocab, load_params
from musicnet.preprocessing.midi_to_wav.utils import MidiConvertedTrack, get_midi_train_ids
from utils import Preprocessor, get_out_dir
import random
import tensorflow as tf
import os
import shutil

params = load_params(["wav_specs_and_notes.*", "midi_to_wav.programs_whitelist"])

out_dir = get_out_dir(params["use_converted_midis"])

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

if params["use_converted_midis"]:
    ids_train_val = get_midi_train_ids()
    TrackClass = MidiConvertedTrack
    if params["programs_whitelist"]:
        instruments_vocab = { (v+1): k for k, v in enumerate(params["programs_whitelist"]) }
else:
    ids_train_val = get_train_ids()
    TrackClass = Track

random.shuffle(ids_train_val)

preprocessor = Preprocessor(**params["preprocessor"], instruments_vocab=instruments_vocab)

class ChunkIterator:
    def __iter__(self):
        self.x_chunks = []
        self.y_chunks = []
        return self

    def __next__(self):
        if (len(self.x_chunks) == 0):
            id = ids_train_val.pop()
            track = TrackClass(id)
            print(f"loaded track {id}")
            x_chunks, y_chunks = preprocessor.preprocess(track)

            self.x_chunks += list(x_chunks)
            self.y_chunks += list(y_chunks)
        return self.x_chunks.pop(0), self.y_chunks.pop(0)
    
total_chunks = 0
for id in ids_train_val:
    track = TrackClass(id)
    total_chunks += preprocessor.count_chunks(track)

print(f"Total chunks: {total_chunks}")

def serialize(x_chunk, y_chunk):
    x_chunk = tf.constant(x_chunk, dtype=tf.float32)
    y_chunk = tf.constant(y_chunk, dtype=tf.float32)
    x = tf.io.serialize_tensor(x_chunk)
    y = tf.io.serialize_tensor(y_chunk)
    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.numpy()])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

chunks = ChunkIterator()
# FIXME:
# We could also do it inside the loop to prevent
# continuation of the train track from entering val set,
# but then the chunks count will get messed up
chunks_iter = iter(chunks)

for ds_type, ds_config in params["datasets"].items():
    print(f"processing {ds_type} set".upper())
    ds_size = ds_config["size"]
    ds_files = ds_config["file_count"]
    if ds_size > 1:
        chunks_per_file = int(ds_size / ds_files)
    else:
        chunks_per_file = int((ds_size * total_chunks) / ds_files)
    print(f"num files: {ds_files}, chunks per file: {chunks_per_file}")
    ds_dir = os.path.join(out_dir, ds_type)
    os.makedirs(ds_dir, 0o775, exist_ok=True)
    for i in range(0, ds_files):
        filepath = os.path.join(ds_dir, f"{str(i).zfill(3)}.tfrecord")
        with tf.io.TFRecordWriter(filepath) as writer:
            for j in range(0, chunks_per_file):
                x_chunk, y_chunk = next(chunks)
                serialized = serialize(x_chunk, y_chunk)
                writer.write(serialized)
        print(f"created {filepath}")
    print("Done!")
    print("\n\n")
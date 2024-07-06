import logging, os

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from musicnet.utils import notes_vocab, instruments_vocab, load_params
import tensorflow as tf
from tensorflow import keras
from glob import glob
from musicnet.model.Transformer import AudioTransformer, TransformerLRSchedule
from tensorflow.nn import weighted_cross_entropy_with_logits
from dvclive.keras import DVCLiveCallback
from pathlib import Path

if len(tf.config.list_physical_devices("GPU")) == 0:
    raise Exception("GPU not found")

params = load_params(("train", "shared"))
target_classes = len(notes_vocab) * len(instruments_vocab)

def decode_record(record_bytes):
    example = tf.io.parse_example(record_bytes, {
        "x": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "y": tf.io.FixedLenFeature([], tf.string, default_value="")
    })
    x = tf.io.parse_tensor(example["x"], tf.float32)
    x.set_shape([None, params["n_mels"]])
    y = tf.io.parse_tensor(example["y"], tf.bool)
    y.set_shape([None, target_classes])

    return (x, y[:-1]), y[1:]

def create_tf_record_ds(source_dir, num_parallel_reads="auto"):
    files = glob(os.path.join(source_dir, "*.tfrecord"))
    if num_parallel_reads == 'auto':
        num_parallel_reads = len(files)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    ds = ds.map(decode_record).shuffle(1000).batch(params["batch_size"]).prefetch(1)
    return ds

train_ds = create_tf_record_ds(str(Path(__file__).parent.with_name("data").joinpath("train")))
val_ds = create_tf_record_ds(str(Path(__file__).parent.with_name("data").joinpath("val")))

model = AudioTransformer(
    n_mels=params["n_mels"],
    d_model=params["d_model"],
    num_layers=params["num_layers"],
    num_heads=params["num_heads"],
    dff=params["dff"],
    seq_len=1000,
    target_classes=target_classes,
    mha_dropout=params["mha_dropout"],
    input_dropout=params["input_dropout"]
)

lr_schedule = TransformerLRSchedule(d_model=params["d_model"])
optimizer = keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

class F1FromSeqLogits(keras.metrics.F1Score):
    def update_state(self, y_true, y_pred, **kwargs):
        y_pred = tf.sigmoid(y_pred)
        y_true = tf.reshape(y_true, [-1, target_classes])
        y_pred = tf.reshape(y_pred, [-1, target_classes])
        return super().update_state(y_true, y_pred, **kwargs)
    
def loss_fn(y_true, y_pred):
    # We use weighted binary cross-entropy with a pos_weight of 300 to account for class imbalance
    return weighted_cross_entropy_with_logits(y_true, y_pred, 300)

model.compile(
    loss=loss_fn,
    optimizer=optimizer,
    metrics=[F1FromSeqLogits(threshold=0.5, average="weighted")]
)

# TODO: Tracking
model.fit(
    train_ds,
    epochs=params["epochs"],
    validation_data=val_ds,
    callbacks=[DVCLiveCallback()]
)

# TODO: Test loading
model.save(str(Path(__file__).with_name("model.keras")))
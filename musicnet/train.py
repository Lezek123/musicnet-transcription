import logging, os

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from musicnet.utils import notes_vocab, instruments_vocab
import tensorflow as tf
from tensorflow import keras
from glob import glob
from musicnet.model.Transformer import AudioTransformer, TransformerLRSchedule
from tensorflow.nn import weighted_cross_entropy_with_logits
from datetime import datetime

if len(tf.config.list_physical_devices("GPU")) == 0:
    raise Exception("GPU not found")

n_mels=128
d_model=128
num_layers=4
num_heads=8
dff=512
target_classes = len(notes_vocab) * len(instruments_vocab)
mha_dropout=0.1
input_dropout=0.1
batch_size=8

def decode_record(record_bytes):
    example = tf.io.parse_example(record_bytes, {
        "x": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "y": tf.io.FixedLenFeature([], tf.string, default_value="")
    })
    x = tf.io.parse_tensor(example["x"], tf.float32)
    x.set_shape([None, n_mels])
    y = tf.io.parse_tensor(example["y"], tf.bool)
    y.set_shape([None, target_classes])

    return (x, y[:-1]), y[1:]

def create_tf_record_ds(source_dir, num_parallel_reads="auto"):
    files = glob(os.path.join(source_dir, "*.tfrecord"))
    if num_parallel_reads == 'auto':
        num_parallel_reads = len(files)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    ds = ds.map(decode_record).shuffle(1000).batch(batch_size).prefetch(1)
    return ds

train_ds = create_tf_record_ds("../data/train")
val_ds = create_tf_record_ds("../data/val")

model = AudioTransformer(
    n_mels=n_mels,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    dff=dff,
    seq_len=1000,
    target_classes=target_classes,
    mha_dropout=mha_dropout,
    input_dropout=input_dropout
)

lr_schedule = TransformerLRSchedule(d_model=d_model)
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
model.fit(train_ds, epochs=100, validation_data=val_ds)

date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# TODO: Test loading
model.save(f"models/{date_str}.keras")
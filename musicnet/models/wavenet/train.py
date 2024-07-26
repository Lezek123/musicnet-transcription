import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from musicnet.utils import notes_vocab, instruments_vocab, load_params, get_training_artifacts_dir, find_lr
import tensorflow as tf
import keras
from musicnet.models.transformer.Transformer import (
    WeightedBinaryCrossentropy,
    F1FromSeqLogits,
)
from musicnet.preprocessing.wav_specs_and_notes.utils import create_tf_record_ds
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from pathlib import Path
import numpy as np

if len(tf.config.list_physical_devices("GPU")) == 0:
    raise Exception("GPU not found")

params = load_params([
    "midi_to_wav.programs_whitelist",
    "wav_specs_and_notes.preprocessor.target_sr",
    "wav_specs_and_notes.preprocessor.chunk_size_sec",
    "wav_specs_and_notes.use_converted_midis",
    "wavenet.*"
])

if params["programs_whitelist"]:
    target_classes = len(notes_vocab) * len(params["programs_whitelist"])
else:
    target_classes = len(notes_vocab) * len(instruments_vocab)

# In this case we don't really have "filters", but raw WAV data, so the filter dimension is just 1
n_filters = 1
ds_params = {
    "architecture": "wavenet",
    "n_filters": n_filters,
    "target_classes": target_classes,
    "batch_size": params["batch_size"],
    "dataset_size": params["dataset_size"],
    "use_converted_midis": params["use_converted_midis"]
}

@tf.function
def calc_positive_class_weight():
    train_ds = create_tf_record_ds("train", **ds_params)
    positive_count = 0.0
    negative_count = 0.0
    for _, y_batch in train_ds:
        positive_count += tf.reduce_sum(y_batch)
        negative_count += tf.reduce_sum(1.0 - y_batch)
    return negative_count / positive_count

pos_class_weight = calc_positive_class_weight().numpy()
loss = WeightedBinaryCrossentropy(pos_class_weight)

sample_memory_bytes = (params["target_sr"] * params["chunk_size_sec"] * 32) / 8
max_cache_memory = 1024 * 1024 * 1024
buffer_size = int(max_cache_memory / sample_memory_bytes)

print("Shuffle buffer size", buffer_size)

train_ds = create_tf_record_ds("train", **ds_params, buffer_size=buffer_size)
val_ds = create_tf_record_ds("val", **ds_params)

def build_model(optimizer, **kwargs):
    model = keras.models.Sequential()
    model.add(tf.keras.Input(shape=[params["target_sr"], 1]))
    # model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5))
    for i, l in enumerate(params["layers"]):
        if i > 0:
            conv_layer = keras.layers.SeparableConv1D
        else:
            conv_layer = keras.layers.Conv1D
        max_x = round(np.log(l["wavenet_kernel"]) / np.log(2)) - 1
        for rate in [2 ** x for x in range(0, max_x + 1)]:
            model.add(conv_layer(
                filters=l["n_maps"],
                kernel_size=2,
                activation=params["activation"],
                dilation_rate=rate,
                padding="same"
            ))
        if l["pooling"]:
            model.add(keras.layers.MaxPool1D(l["pooling"]))
    model.add(keras.layers.Conv1D(target_classes, kernel_size=1, activation=None))
    model.compile(
        loss=WeightedBinaryCrossentropy(pos_class_weight),
        optimizer=optimizer,
        **kwargs
    )
    return model

model = build_model(keras.optimizers.Adam())
model.summary()

for batch in train_ds:
    print("X shape", batch[0].shape, "Y shape", batch[1].shape)
    break

model_path, live_path = get_training_artifacts_dir(Path(__file__))    

with Live(live_path) as live:
    metrics = [
        F1FromSeqLogits(threshold=0.5, average="weighted", name="f1_weighted"),
        F1FromSeqLogits(threshold=0.5, average="micro", name="f1_global"),
        keras.metrics.Precision(0, name="precision"),
        keras.metrics.Recall(0, name="recall")
    ]
    if params["lr"] == "auto":
        model, best_lr, init_epoch = find_lr(build_model, train_ds)
        model.compile(optimizer=keras.optimizers.Adam(best_lr), loss=loss, metrics=metrics)
        live.log_param("lr", best_lr)
    else:
        model = build_model(optimizer=keras.optimizers.Adam(params["lr"]), metrics=metrics)
        init_epoch = 0
        live.log_param("lr", params["lr"])
    
    model.fit(
        train_ds,
        epochs=params["epochs"],
        initial_epoch=init_epoch,
        validation_data=val_ds,
        callbacks=[DVCLiveCallback(live=live)],
    )

model.save(model_path)

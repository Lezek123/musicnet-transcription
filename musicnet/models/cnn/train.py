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

if len(tf.config.list_physical_devices("GPU")) == 0:
    raise Exception("GPU not found")

params = load_params([
    "midi_to_wav.programs_whitelist",
    "wav_specs_and_notes.preprocessor.spectogram.n_filters",
    "wav_specs_and_notes.use_converted_midis",
    "cnn.*"
])

if params["programs_whitelist"]:
    target_classes = len(notes_vocab) * len(params["programs_whitelist"])
else:
    target_classes = len(notes_vocab) * len(instruments_vocab)

ds_params = {
    "architecture": "cnn",
    "n_filters": params["n_filters"],
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
loss=WeightedBinaryCrossentropy(pos_class_weight)

train_ds = create_tf_record_ds("train", **ds_params)
val_ds = create_tf_record_ds("val", **ds_params)

def build_model(optimizer, **kwargs):
    if params["conv_type"] == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    inputs = tf.keras.Input(shape=[1000, params["n_filters"]])
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        input_shape=[1000, params["n_filters"]]
    )(inputs)

    skip_x = None
    for l in range(params["n_layers"]):
        x = conv_layer(
            params["n_neurons"],
            kernel_size=params["kernel_size"],
            padding="same",
            activation=None
        )(x)
        # x = tf.keras.layers.BatchNormalization(
        #     epsilon=1e-5,
        #     input_shape=[1000, params["n_neurons"]]
        # )(x)
        if l % 2 == 0 and skip_x is not None:
            # Residual connection
            x = tf.keras.layers.Add()([x, skip_x])
            x = tf.keras.activations.get(params["activation"])(x)
            skip_x = x
        else:
            x = tf.keras.activations.get(params["activation"])(x)
            if skip_x is None:
                skip_x = x

    if params["dropout_rate"]:
        x = tf.keras.layers.Dropout(
            rate=params["dropout_rate"],
            # Force the same channels to be dropped in all timesteps 
            # see: https://keras.io/api/layers/regularization_layers/dropout/
            noise_shape=(None, 1, params["n_neurons"])
        )(x)
    x = conv_layer(target_classes, kernel_size=params["kernel_size"], padding="same", activation=None)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        **kwargs
    )
    return model

# model = build_model(keras.optimizers.Adam())
# model.summary()
# tf.keras.utils.plot_model(model, "model.png")

model_path, live_path = get_training_artifacts_dir(Path(__file__))    

with Live(live_path) as live:
    metrics = [F1FromSeqLogits(threshold=0.5, average="weighted")]
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

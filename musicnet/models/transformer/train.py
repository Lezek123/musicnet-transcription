import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from musicnet.utils import notes_vocab, instruments_vocab, load_params, get_training_artifacts_dir
import tensorflow as tf
import keras
from musicnet.models.transformer.Transformer import (
    AudioTransformer,
    WarmupLRSchedule,
    WeightedBinaryCrossentropy,
    F1FromSeqLogits,
    EncoderOnlyAudioTransformer,
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
    "transformer.*"
])

if params["programs_whitelist"]:
    target_classes = len(notes_vocab) * len(params["programs_whitelist"])
else:
    target_classes = len(notes_vocab) * len(instruments_vocab)

ds_params = {
    "architecture": params["architecture"],
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

train_ds = create_tf_record_ds("train", **ds_params)
val_ds = create_tf_record_ds("val", **ds_params)

lr_schedule = WarmupLRSchedule(max_lr=params["max_lr"], warmup_steps=params["warmup_steps"])
optimizer = keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model_params = {
    "n_filters": params["n_filters"],
    "d_model": params["d_model"],
    "num_layers": params["num_layers"],
    "num_heads": params["num_heads"],
    "dff": params["dff"],
    "seq_len": 1000,
    "target_classes": target_classes,
    "mha_dropout": params["mha_dropout"],
    "input_dropout": params["input_dropout"],
}

if params["architecture"] == "encoder-only":
    model = EncoderOnlyAudioTransformer(**model_params)
else:
    model = AudioTransformer(**model_params)

for batch in train_ds:
    print(batch[0].shape, batch[1].shape)
    break

model.compile(
    loss=WeightedBinaryCrossentropy(pos_class_weight),
    optimizer=optimizer,
    metrics=[F1FromSeqLogits(threshold=0.5, average="weighted")],
)

model_path, live_path = get_training_artifacts_dir(Path(__file__))

with Live(live_path) as live:
    model.fit(
        train_ds,
        epochs=params["epochs"],
        validation_data=val_ds,
        callbacks=[DVCLiveCallback(live=live)],
    )

model.save(model_path)

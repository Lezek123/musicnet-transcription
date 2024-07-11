import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from musicnet.utils import (
    notes_vocab,
    instruments_vocab,
    load_params,
    create_tf_record_ds,
)
import tensorflow as tf
from tensorflow import keras
from musicnet.model.Transformer import (
    AudioTransformer,
    WarmupLRSchedule,
    loss_fn,
    F1FromSeqLogits,
    EncoderOnlyAudioTransformer,
)
from dvclive.keras import DVCLiveCallback
from pathlib import Path

if len(tf.config.list_physical_devices("GPU")) == 0:
    raise Exception("GPU not found")

params = load_params(("train", "shared"))
target_classes = len(notes_vocab) * len(instruments_vocab)

train_ds = create_tf_record_ds(
    "train", params["n_mels"], target_classes, params["batch_size"]
)
val_ds = create_tf_record_ds(
    "val", params["n_mels"], target_classes, params["batch_size"]
)

model_params = {
    "n_mels": params["n_mels"],
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
    print("\n\nUsing encoder-only architecture\n\n".upper())
    train_ds = train_ds.map(lambda *batch: (batch[0][0], batch[1][:, :-1, :]))
    val_ds = val_ds.map(lambda *batch: (batch[0][0], batch[1][:, :-1, :]))
    model = EncoderOnlyAudioTransformer(**model_params)
else:
    model = AudioTransformer(**model_params)

lr_schedule = WarmupLRSchedule(max_lr=params["max_lr"], warmup_steps=params["warmup_steps"])
optimizer = keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model.compile(
    loss=loss_fn,
    optimizer=optimizer,
    metrics=[F1FromSeqLogits(threshold=0.5, average="weighted")],
)

model.fit(
    train_ds,
    epochs=params["epochs"],
    validation_data=val_ds,
    callbacks=[DVCLiveCallback()],
)

# TODO: Test loading
model.save(str(Path(__file__).with_name("model.keras")))

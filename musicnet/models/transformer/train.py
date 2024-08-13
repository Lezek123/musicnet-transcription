from musicnet.models.utils import (
    calc_positive_class_weight,
    get_common_metrics,
    WeightedBinaryCrossentropy
)
import tensorflow as tf
import keras
from musicnet.models.transformer.Transformer import (
    AudioTransformer,
    WarmupLRSchedule,
    EncoderOnlyAudioTransformer,
)
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from musicnet.config.model import TransformerConfig, TransformerArchitecture

def train(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, config: TransformerConfig, live: Live, model_path: str):
    pos_class_weight = calc_positive_class_weight(train_ds).numpy()
    loss = WeightedBinaryCrossentropy(pos_class_weight)

    lr_schedule = WarmupLRSchedule(max_lr=config.max_lr, warmup_steps=config.warmup_steps)
    optimizer = keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_iter = iter(train_ds)
    (x_batch, y_batch) = train_iter.get_next()
    x_shape = x_batch.shape[1:]
    y_shape = y_batch.shape[1:]

    print("X shape", x_shape, "Y shape", y_shape)

    model_params = {
        "n_filters": x_shape[-1],
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "dff": config.dff,
        "seq_len": 1000, # TODO: Calculate
        "target_classes": y_shape[-1],
        "mha_dropout": config.mha_dropout,
        "input_dropout": config.input_dropout,
    }

    if config.architecture == TransformerArchitecture.ENCODER:
        model = EncoderOnlyAudioTransformer(**model_params)
    else:
        train_ds = train_ds.map(lambda x_batch, y_batch: (x_batch[:, :, -1], y_batch[:, :, -1]), y_batch[1, :, :])
        model = AudioTransformer(**model_params)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=get_common_metrics(),
    )

    model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        callbacks=[DVCLiveCallback(live=live)],
    )
    model.save(model_path)
    live.log_artifact(model_path)

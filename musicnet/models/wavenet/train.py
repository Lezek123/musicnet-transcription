from musicnet.models.utils import (
    WeightedBinaryCrossentropy,
    calc_positive_class_weight,
    find_lr,
    get_common_metrics
)
from musicnet.config.model.WaveNetConfig import WaveNetConfig
from musicnet.config.model.common import LRDerivation
import tensorflow as tf
import keras
from dvclive import Live
from dvclive.keras import DVCLiveCallback
import numpy as np

def train(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, config: WaveNetConfig, live: Live, model_path: str):
    pos_class_weight = calc_positive_class_weight(train_ds)
    loss = WeightedBinaryCrossentropy(pos_class_weight)

    train_iter = iter(train_ds)
    (x_batch, y_batch) = train_iter.get_next()
    x_shape = x_batch.shape[1:]
    y_shape = y_batch.shape[1:]
    print("X shape", x_shape, "Y shape", y_shape)

    def build_model(optimizer, **kwargs):
        model = keras.models.Sequential()
        model.add(tf.keras.Input(shape=x_shape))
        # model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5))
        for i, l in enumerate(config.layers):
            if i > 0:
                conv_layer = keras.layers.SeparableConv1D
            else:
                conv_layer = keras.layers.Conv1D
            max_x = round(np.log(l.wavenet_kernel) / np.log(2)) - 1
            for rate in [2 ** x for x in range(0, max_x + 1)]:
                model.add(conv_layer(
                    filters=l.n_maps,
                    kernel_size=2,
                    activation=config.activation.value,
                    dilation_rate=rate,
                    padding="same"
                ))
            if l.pooling:
                model.add(keras.layers.MaxPool1D(l.pooling))
        model.add(keras.layers.Conv1D(y_shape[-1], kernel_size=1, activation=None))
        model.compile(
            loss=WeightedBinaryCrossentropy(pos_class_weight),
            optimizer=optimizer,
            **kwargs
        )
        return model

    model = build_model(keras.optimizers.Adam())
    model.summary()

    metrics = get_common_metrics()
    if config.lr == LRDerivation.AUTO:
        model, best_lr, init_epoch = find_lr(build_model, train_ds)
        model.compile(optimizer=keras.optimizers.Adam(best_lr), loss=loss, metrics=metrics)
        live.log_param("lr", best_lr)
    else:
        model = build_model(optimizer=keras.optimizers.Adam(config.lr), metrics=metrics)
        init_epoch = 0
        live.log_param("lr", config.lr)
    
    model.fit(
        train_ds,
        epochs=config.epochs,
        initial_epoch=init_epoch,
        validation_data=val_ds,
        callbacks=[DVCLiveCallback(live=live)],
    )

    model.save(model_path)
    live.log_artifact(model_path)

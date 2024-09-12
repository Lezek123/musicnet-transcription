import tensorflow as tf
import keras
from musicnet.models.utils import (
    WeightedBinaryCrossentropy,
    calc_positive_class_weight,
    find_lr,
    get_common_metrics,
    get_common_callbacks,
    restore_checkpoint_model
)
from dvclive import Live
from musicnet.config.model.CNNConfig import CNNConfig, ConvType
from musicnet.config.model.common import LRDerivation

# TODO:
# Configure DVC to always use temp dir for experiments
# Try to understand what it is actually doing w.r.t. stashing, refs etc.
# Perhaps some other options like --include-utracked will be needed to get the behavior I'm expecting
# (but in general avoid running experiments with untracked files for other purposes than simple tests and delete them afterwards)

def train(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, config: CNNConfig, live: Live, model_path: str):
    pos_class_weight = calc_positive_class_weight(train_ds).numpy()
    loss = WeightedBinaryCrossentropy(pos_class_weight)

    print("Pos class weight:", pos_class_weight)
    
    train_iter = iter(train_ds)
    (x_batch, y_batch) = train_iter.get_next()
    x_shape = x_batch.shape[1:]
    y_shape = y_batch.shape[1:]

    def build_model(optimizer, **kwargs):
        if config.conv_type == ConvType.SEPARABLE:
            conv_layer = tf.keras.layers.SeparableConv1D
        else:
            conv_layer = tf.keras.layers.Conv1D

        inputs = tf.keras.Input(shape=x_shape)
        x = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            input_shape=x_shape[1:]
        )(inputs)

        skip_x = None
        for l in range(config.n_layers):
            x = conv_layer(
                config.n_neurons,
                kernel_size=config.kernel_size,
                padding="same",
                activation=None
            )(x)
            if l % 2 == 0 and skip_x is not None:
                # Residual connection
                x = tf.keras.layers.Add()([x, skip_x])
                x = tf.keras.layers.Activation(config.activation.value)(x)
                skip_x = x
            else:
                x = tf.keras.layers.Activation(config.activation.value)(x)
                if skip_x is None:
                    skip_x = x

        if config.dropout_rate:
            x = tf.keras.layers.Dropout(
                rate=config.dropout_rate,
                # Force the same channels to be dropped in all timesteps 
                # see: https://keras.io/api/layers/regularization_layers/dropout/
                noise_shape=(None, 1, config.n_neurons)
            )(x)
        x = conv_layer(y_shape[-1], kernel_size=config.kernel_size, padding="same", activation=None)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(
            loss=loss,
            optimizer=optimizer,
            **kwargs
        )
        return model
    
    model, init_epoch = restore_checkpoint_model()
    if model is None:
        # No checkpoint model found: build a clean model instance
        metrics = get_common_metrics()
        if config.lr == LRDerivation.AUTO:
            model, best_lr, init_epoch = find_lr(build_model, train_ds)
            for _ in range(init_epoch):
                live.next_step()
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
        callbacks=get_common_callbacks(live),
    )

    model.save(model_path)
    live.log_artifact(model_path)

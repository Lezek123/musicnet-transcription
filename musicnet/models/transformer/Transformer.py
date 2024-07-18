# Defines a Transformer seq2seq architecture model similar to OpenAI's Whisper
# Ref: https://github.com/openai/whisper

import numpy as np
import tensorflow as tf
import keras
from tensorflow.nn import weighted_cross_entropy_with_logits

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.max_dims = max_dims
        if max_dims % 2 == 1:
            max_dims += 1  # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))

    def call(self, inputs):
        shape = tf.shape(inputs)
        scale_factor = tf.math.sqrt(tf.cast(self.max_dims, tf.float32))
        return inputs * scale_factor + self.positional_embedding[:, : shape[-2], : shape[-1]]


class BaseAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, mha_dropout=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=mha_dropout
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(keras.layers.Layer):
    def __init__(
        self,
        # Dimensionality of encoder input/output vectors
        d_model,
        # Number of neurons in the hidden feed-forward layer
        dff,
        # Max sequence length
        seq_len,
        # Number of attention heads
        num_heads,
        # Number of encoder layers
        num_layers,
        # Dropout rate for the input embeddings
        input_dropout,
        # Dropout rate for the MHA layers
        mha_dropout,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dropout = input_dropout
        self.pos_enc = PositionalEncoding(max_steps=seq_len, max_dims=d_model)
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, mha_dropout=mha_dropout
            )
            for _ in range(num_layers)
        ]
        self.dropout = keras.layers.Dropout(input_dropout)

    def call(self, x):
        x = self.pos_enc(x)
        if self.input_dropout > 0:
            x = self.dropout(x)
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, mha_dropout):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=mha_dropout
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=mha_dropout
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(keras.layers.Layer):
    def __init__(
        self,
        *,
        # Number of decoder layers
        num_layers,
        # Dimensionality of decoder input/output vectors
        d_model,
        # Number of attention heads in each decoder layer
        num_heads,
        # Number of neurons in the hidden dense layer in "FeedForward" component
        dff,
        # Max sequence length
        seq_len,
        # Dropout rate for the input embeddings
        input_dropout,
        # Dropout rate for the MHA layers
        mha_dropout
    ):
        super(Decoder, self).__init__()
        self.input_dropout = input_dropout
        self.pos_enc = PositionalEncoding(max_dims=d_model, max_steps=seq_len)
        self.dropout = keras.layers.Dropout(input_dropout)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, mha_dropout=mha_dropout
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_enc(x)  # (batch_size, target_seq_len, d_model)
        if self.input_dropout > 0:
            x = self.dropout(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(keras.layers.Layer):
    def __init__(
        self,
        *,
        # Number of stacked encoder / decoder layers
        num_layers,
        # Dimensionality of the MHA layers' input and output vectors
        d_model,
        # Number of attention heads in each MHA layer
        num_heads,
        # Number of neurons in the hidden dense layer of the FeedForward component
        dff,
        # Maximum encoder / decoder sequence length
        seq_len,
        # Number of classes in Y
        target_classes,
        # MHA dropout
        mha_dropout,
        # Input dropout
        input_dropout
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            seq_len=seq_len,
            mha_dropout=mha_dropout,
            input_dropout=input_dropout,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            seq_len=seq_len,
            mha_dropout=mha_dropout,
            input_dropout=input_dropout,
        )

        self.final_layer = tf.keras.layers.Dense(target_classes)

    def call(self, inputs):
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


class AudioTransformer(keras.Model):
    def __init__(
        self,
        *,
        n_filters,
        # Dimensionality of the MHA layers' input and output vectors
        d_model,
        # Number of classes in Y
        target_classes,
        **kwargs
    ):
        super().__init__()
        self.config_parmas = { "n_filters": n_filters, "d_model": d_model, "target_classes": target_classes, **kwargs }
        self.conv1 = keras.layers.Conv1D(
            d_model, kernel_size=3, padding="same", input_shape=[None, n_filters]
        )
        self.conv2 = keras.layers.Conv1D(d_model, kernel_size=3, strides=2)
        self.embedding = keras.layers.Dense(d_model, input_shape=[None, target_classes])
        self.transformer = Transformer(
            d_model=d_model, target_classes=target_classes, **kwargs
        )

    def get_config(self):
        return self.config_parmas

    def call(self, inputs):
        # print(tf.shape(inputs))
        # raise Exception("Test")
        mel_specs, notes = inputs

        context = self.conv1(mel_specs)
        context = self.conv2(mel_specs)

        x = self.embedding(notes)

        logits = self.transformer((context, x))

        return logits
    
class EncoderOnlyAudioTransformer(keras.Model):
    def __init__(
        self,
        *,
        n_filters,
        # Dimensionality of the MHA layers' input and output vectors
        d_model,
        # Number of classes in Y
        target_classes,
        # Number of encoder layers
        num_layers,
        # Number of attention heads in each encoder layer
        num_heads,
        # Number of neurons in the feed forward layers
        dff,
        # Encoder sequence length
        seq_len,
        # MHA layers dropout rate
        mha_dropout,
        # Pre-encoder dropout rate
        input_dropout,
        **kwargs
    ):
        super().__init__()
        self.config_parmas = {
            "n_filters": n_filters,
            "d_model": d_model,
            "target_classes": target_classes,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dff": dff,
            "seq_len": seq_len,
            "mha_dropout": mha_dropout,
            "input_dropout": input_dropout,
            **kwargs
        }
        self.norm = keras.layers.BatchNormalization(epsilon=1e-5)
        self.conv1 = keras.layers.Conv1D(
            d_model, kernel_size=3, padding="same", input_shape=[None, n_filters]
        )
        self.conv2 = keras.layers.Conv1D(d_model, kernel_size=3, padding="same")
        self.embedding = keras.layers.Dense(d_model, input_shape=[None, target_classes])
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            seq_len=seq_len,
            mha_dropout=mha_dropout,
            input_dropout=input_dropout,
        )
        self.final_layer = tf.keras.layers.Dense(target_classes)
    
    def get_config(self):
        return self.config_parmas

    def call(self, mel_spectograms):
        x = self.norm(mel_spectograms)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.encoder(x)
        logits = self.final_layer(x)

        return logits


# Use the Adam optimizer with a custom learning rate scheduler according to the formula in the original Transformer paper.
class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.config_params = { "d_model": d_model, "warmup_steps": warmup_steps }

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        return self.config_params

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
# Custom warmup LR schedule with max_lr
class WarmupLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps=4000):
        super().__init__()

        self.config_params = { "max_lr": max_lr, "warmup_steps": warmup_steps }

        self.max_lr = tf.cast(max_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def get_config(self):
        return self.config_params

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.sqrt(self.warmup_steps) / tf.math.sqrt(step)
        arg2 = step * (self.warmup_steps**-1)

        return self.max_lr * tf.math.minimum(arg1, arg2)

class F1FromSeqLogits(keras.metrics.F1Score):
    def update_state(self, y_true, y_pred, **kwargs):
        y_pred = tf.sigmoid(y_pred)
        y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        return super().update_state(y_true, y_pred, **kwargs)
    
class WeightedBinaryCrossentropy(keras.losses.Loss):
    def __init__(self, pos_weight, name="weighted_binary_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = pos_weight

    def get_config(self):
        return { **super().get_config(), "pos_weight": self.pos_weight }

    def call(self, y_true, y_pred):
        return weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight)
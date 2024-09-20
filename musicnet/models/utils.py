import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.nn import weighted_cross_entropy_with_logits # type: ignore
from pathlib import Path
from tqdm import tqdm
from dvclive.keras import DVCLiveCallback
from musicnet.utils import IS_CLOUD

if IS_CLOUD:
    MODEL_PATH = "/gcs/musicnet-job-data/model.keras"
    CHECKPOINT_DIR = "/gcs/musicnet-job-data/model-checkpoint"
else:
    MODEL_PATH = str(Path(__file__).with_name("model.keras"))
    CHECKPOINT_DIR = str(Path(__file__).with_name("checkpoint"))

CHECKPOINT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.keras")
CHECKPOINT_EPOCH_PATH = os.path.join(CHECKPOINT_DIR, "epoch")

class SaveEpochCallback(keras.callbacks.Callback):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch: int, logs=None):
        with open(self.file_path, "w") as state_file:
            # Using `epoch + 1`, because this will be the initial epoch on the resumed run 
            state_file.write(str(epoch + 1))

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

@tf.function
def calc_positive_class_weight(train_ds) -> tf.Tensor:
    positive_count = 0.0
    negative_count = 0.0
    for _, y_batch in train_ds:
        positive_count += tf.reduce_sum(y_batch)
        negative_count += tf.reduce_sum(1.0 - y_batch)
    return negative_count / positive_count

def get_common_metrics():
    return [
        F1FromSeqLogits(threshold=0.5, average="weighted", name="f1_weighted"),
        F1FromSeqLogits(threshold=0.5, average="micro", name="f1_global"),
        keras.metrics.Precision(0, name="precision"),
        keras.metrics.Recall(0, name="recall")
    ]

def get_common_callbacks(live):
    return [
        DVCLiveCallback(live=live),
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_MODEL_PATH),
        SaveEpochCallback(CHECKPOINT_EPOCH_PATH)
    ]

def restore_checkpoint_model():
    if os.path.exists(CHECKPOINT_EPOCH_PATH):
        with open(CHECKPOINT_EPOCH_PATH, "r") as state_file:
            initial_epoch = int(state_file.readline())
        if os.path.exists(CHECKPOINT_MODEL_PATH):
            model = keras.models.load_model(CHECKPOINT_MODEL_PATH)
            return model, initial_epoch
    return None, 0

def find_lr(build_model, x, y=None, early_stopping=True):
    print("Searching for best learning rate...")
    all_lrs = np.array([1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1])
    models = {}
    epochs = [1, 2, 4]
    n_lrs = [10, 5, 3, 1]

    lrs = all_lrs.copy()
    for i, e in enumerate(epochs):
        initial_epoch = 0 if i == 0 else epochs[i-1]
        performances = []
        lrs_left = n_lrs[i]
        lrs = np.sort(lrs[:lrs_left])
        if len(models):
            models = { lr: models[lr] for lr in lrs }
        for lr in tqdm(list(lrs)):
            if lr not in models:
                model = build_model(keras.optimizers.Adam(lr))
                models[lr] = model
            history = models[lr].fit(x=x, y=y, epochs=e, initial_epoch=initial_epoch, verbose=0)
            loss = history.history["loss"][-1]
            performances.append(loss)
            print(lr, loss)
            if early_stopping and len(performances) > n_lrs[i+1]:
                if loss > max(np.sort(performances)[:n_lrs[i+1]]):
                    print("Early stopping activated")
                    break
        print("\n\n")
        lrs = lrs[np.argsort(performances)]
    best_lr = lrs[0]
    print("LR found: ", best_lr)
    return models[best_lr], float(best_lr), epochs[-1]
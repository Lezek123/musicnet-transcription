import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.nn import weighted_cross_entropy_with_logits # type: ignore
from musicnet.utils import IS_CLOUD, PROJECT_ROOT_DIR
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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

# TODO: Setup a remote in the Google Bucket for DVC instead and then pull from there
def get_training_artifacts_dir():
    if IS_CLOUD:
        date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_path = f"/gcs/musicnet-ds/jobs/{date_str}"
        model_path = os.path.join(base_path, "model.keras")
        live_path = os.path.join(base_path, "dvclive")
    else:
        model_path = str(Path(__file__).with_name("model.keras"))
        live_path = os.path.join(PROJECT_ROOT_DIR, "dvclive")
    return model_path, live_path

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
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from dvclive import Live
from tensorflow import keras
from musicnet.config import Config
from musicnet.preprocessing.wav_chunks_tfrecord.utils import create_tf_record_ds
from musicnet.preprocessing.utils import get_datasets_info, load_vocabs
from musicnet.preprocessing.dataset.base import DsName
from dataclasses import dataclass
from .utils import MODEL_PATH, F1FromSeqLogits

# FIXME: Obecne wykresy/metryki wskazują, że ilość błędów dla zbioru testowego jest niewspółmiernie wysoka
# Potencjalne przyczyny:
# - labele dla zbioru testowego nie są poprawne:
#   - pliki midi nie są poprawnie przetwarzane przez midi_to_wav (np. filtrowanie instrumentów nie działa poprawnie)
#   - używany jest inny instruments_vocab do przetwarzania różnych zbiorów danych
# - dane w zbiorze testowym naturalnie znacząco różnią się od zbioru walidacyjnego:
#   - model gubi się z powodu częstego występowania b. wielu dźwięków jednocześnie (do 9) lub b. krótkich dźwięków
#   - model ma problem gdy grane są jednocześnie dźwięki mocno od siebie oddalone (z powodu niskiego notes_std)
#   - model ma problem z powodu wartości velocity różnych od tych na których był trenowany
# 
#   Możliwe rozwiązania:
#   - zwiększyć max_silmultaneous_notes i notes_std w zbiorze walidacyjnym (ew. poeksperymentować z velocity) i porównać wyniki.
#     Jeżeli zaczną się zbliżać do tych ze zbioru testowego to prawdopodobnie mamy przyczynę i można spróbować zmienić zbiór treningowy.  
# 
# - zagadka: dlaczego w przypadku ciszy (silmultaneous_notes: 0) mamy średnie FP w wysokości aż 2.6 na zbiorze testowym,
#   podczas gdy na walidacyjnym wynosi tylko niecałe 0.7?
# 
# Update 29-08-2024: Jednym z gł. przyczyn było prawdopodobnie niepoprawne przetwarzanie plików midi w funkcji get_midi_notes
# (niewłaściwe aktualizowanie current_time)

class AverageFpByLabelsCount(tf.keras.metrics.Metric):
    def __init__(self, max_labels: int, threshold=0., name='avg_fp_by_lables_count', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.max_labels = max_labels
        # [label_count, [indices_count, fp_count]] for each label_count between 0 and max_labels
        self.counters = self.add_weight(
            shape=(max_labels + 1, 2),
            initializer='zeros',
            name='counters',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1, y_true.shape[-1]))
        y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))
        for s in range(0, self.max_labels + 1):
            indices = tf.reduce_sum(tf.cast(y_true, tf.int32), axis=1) == s
            indices_count = tf.reduce_sum(tf.cast(indices, tf.float32))
            fp_count = tf.reduce_sum(tf.cast((y_true[indices] == False) & (y_pred[indices] >= self.threshold), tf.float32))
            self.counters[s, :].assign(self.counters[s, :] + [indices_count, fp_count])

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))

    def result(self):
        return self.counters[:, 1] / self.counters[:, 0]
    
class FpConfMatrix(tf.keras.metrics.Metric):
    def __init__(self, threshold = 0., name='f1_conf_matrix', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.built = False

    def build(self, y_true_shape):
        # Shape: (num_classes, num_classes, 2)
        # [true_label, pred_label, 0] is the total count of examples in y_true for which example[true_label] == True and example[pred_label] == False
        # [true_label, pred_label, 1] is the total count of pred_label false positives for given true_label
        self.counters = self.add_weight(
            shape=(y_true_shape[-1], y_true_shape[-1], 2),
            initializer='zeros',
            name='counters',
            dtype=tf.float32
        )
        self.built = True
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true.shape)
        y_true = tf.reshape(y_true, (-1, y_true.shape[-1]))
        y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))
        for i in range(0, y_true.shape[-1]):
            indices = y_true[:, i] == True
            # Shape: (num_classes,)
            total_count = tf.reduce_sum(
                tf.cast(y_true[indices] == False, tf.float32),
                axis=0
            )
            fp_count = tf.reduce_sum(
                tf.cast((y_pred[indices] >= self.threshold) & (y_true[indices] == False), tf.float32),
                axis=0
            )
            self.counters[i, :, 0].assign(self.counters[i, :, 0] + total_count) 
            self.counters[i, :, 1].assign(self.counters[i, :, 1] + fp_count)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))
        
    def result(self):
        return self.counters[:, :, 1] / self.counters[:, :, 0]
    
# Calculates total count of true_positives, false_positives and false_negatives for each class
class PerClassStats(tf.keras.metrics.Metric):
    def __init__(self, threshold = 0., name='per_class_stats', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.built = False

    def build(self, y_true_shape):
        # Shape: (3, num_classes)
        # [i, class_idx] is TP for i=0, FP for i=1, FN for i=2
        self.counters = self.add_weight(
            shape=(3, y_true_shape[-1]),
            initializer='zeros',
            name='counters',
            dtype=tf.int32
        )
        self.built = True
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true.shape)
        y_true = tf.reshape(y_true, (-1, y_true.shape[-1]))
        y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))
        # True positives
        self.counters[0].assign(
            self.counters[0] + tf.reduce_sum(tf.cast((y_true == True) & (y_pred >= self.threshold), tf.int32), axis=0)
        )
        # False positives
        self.counters[1].assign(
            self.counters[1] + tf.reduce_sum(tf.cast((y_true == False) & (y_pred >= self.threshold), tf.int32), axis=0)
        )
        # False negatives
        self.counters[2].assign(
            self.counters[2] + tf.reduce_sum(tf.cast((y_true == True) & (y_pred < self.threshold), tf.int32), axis=0)
        )

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))
        
    def result(self):
        return self.counters

@dataclass
class PlotsLogger:
    ds_name: str
    live: Live
    notes_vocab: dict[int, int]

    def log_f1_by_class_plot(self, data: np.ndarray):
        df = pd.DataFrame({
            "note": self.notes_vocab.keys(),
            "f1_score": data
        })
        self.live.log_plot(
            f"{self.ds_name}/f1_by_class",
            df,
            x="f1_score",
            y="note",
            template="bar_horizontal",
            title=f"{self.ds_name.upper()}: F1 score by note"
        )

    def log_average_fp_count_by_silmultaneous_notes_count(self, data: np.ndarray):
        df = pd.DataFrame({
            "silmultaneous_notes": list(range(0, data.shape[-1])),
            "avg_fp": data
        })

        self.live.log_plot(
            f"{self.ds_name}/avg_fp_count_by_silmultaneous_notes_count",
            df,
            x="avg_fp",
            y="silmultaneous_notes",
            template="bar_horizontal",
            title=f"{self.ds_name.upper()}: Avg. false positives count by silmultaneous notes count"
        )

    def log_fp_confusion_matrix_plot(self, data: np.ndarray):
        # For each note we have a percentage of false positives of other notes
        # We convert it to mock_y_true and mock_y_pred values in order to make use of
        # live.log_sklearn_plot
        mock_y_true = []
        mock_y_pred = []
        precision = 3
        for tn, tk in self.notes_vocab.items():
            for pn, pk in self.notes_vocab.items():
                count = int(data[tk][pk] * 10 ** precision)
                mock_y_true += count * [tn]
                mock_y_pred += count * [pn]
        
        self.live.log_sklearn_plot(
            "confusion_matrix",
            mock_y_true,
            mock_y_pred,
            name=f"{self.ds_name}/fp_confusion_matrix",
            title=f"{self.ds_name.upper()}: Confusion matrix of false positives (unit: 1/{10**precision})"
        )

    def tp_fp_fn_by_note_plot(self, data):
        counts = pd.DataFrame({
            "note": list(self.notes_vocab.keys()) * 3,
            "metric": np.repeat(["tp", "fp", "fn"], len(self.notes_vocab)),
            "value": np.concatenate(data, axis=0)
        })

        fig = plt.figure(figsize=(10, 25))
        sns.barplot(counts, x="value", y="note", hue="metric", orient="y", ax=plt.gca())
        self.live.log_image(f"{self.ds_name}/tp_fp_fn_by_note_plot.png", fig)

def eval(config: Config, live: Live) -> None:
    ds_infos = get_datasets_info(config)

    datasets: dict[DsName, tf.data.Dataset] = {}
    for ds_info in ds_infos:
        if ds_info.name in ["val", "test"]:
            datasets[ds_info.name] = create_tf_record_ds(ds_info.config, ds_info.name, shuffle=False)

    model = keras.models.load_model(MODEL_PATH)

    _, notes_vocab = load_vocabs(config)
    
    # Log test metrics
    metric_values = model.evaluate(datasets["test"])
    for i, metric_value in enumerate(metric_values):
        metric_name = model.metrics_names[i]
        live.log_metric(f"test/{metric_name}", metric_value, plot=False)

    max_labels = np.array([y.sum(axis=-1).max() for ds in datasets.values() for _, y in ds.as_numpy_iterator()], dtype=np.int32).max()
    plot_metrics = [
        F1FromSeqLogits(average=None, threshold=0.5),
        AverageFpByLabelsCount(max_labels=max_labels),
        FpConfMatrix(),
        PerClassStats()
    ]
    model.compile(metrics=plot_metrics)

    for ds_name, ds in datasets.items():
        _, f1_by_class, avg_fp_by_labels_count, fp_conf_matrix, per_class_stats = model.evaluate(ds)

        logger = PlotsLogger(ds_name, live, notes_vocab)

        logger.log_f1_by_class_plot(np.nan_to_num(f1_by_class))
        logger.log_average_fp_count_by_silmultaneous_notes_count(np.nan_to_num(avg_fp_by_labels_count))
        logger.log_fp_confusion_matrix_plot(np.nan_to_num(fp_conf_matrix))
        logger.tp_fp_fn_by_note_plot(np.nan_to_num(per_class_stats))
    
    # TODO: Establish what kind of metrics I want to have here:
    # - bar plot of precision/f1 grouped by individual notes
    # - bar plot of precision/f1 grouped by numer of silmulataneous notes
    # - bar plot of precision/f1 grouped by note_length bins
    # - bar plot of precision/f1 grouped by time_playing / time_remaining bins
    # - If midi is a source: bar plot of precision/f1 grouped by velocity bins (like 30-40, 40-50, 50-60 etc.) 
    # - visualization of some fragment of predictions vs true values (something like notebooks/utils:y_vs_y_pred_vis)
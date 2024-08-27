import numpy as np
import tensorflow as tf
import pandas as pd
from dvclive import Live
from tensorflow import keras
from musicnet.config import to_config_object, Config
from musicnet.preprocessing.wav_chunks_tfrecord.utils import create_tf_record_ds
from musicnet.preprocessing.utils import get_datasets_info, load_vocabs
from musicnet.preprocessing.dataset.base import DsName
from dataclasses import dataclass
from .utils import MODEL_PATH

@dataclass
class PlotsLogger:
    y_true: np.ndarray
    y_pred: np.ndarray
    live: Live
    notes_vocab: dict[int, int]
    

    def log_f1_by_class_plot(self):
        f1_per_class_metric = keras.metrics.F1Score(average=None, threshold=0.5)
        f1_per_class = f1_per_class_metric(self.y_true, self.y_pred).numpy()
        df = pd.DataFrame({
            "note": self.notes_vocab.keys(),
            "f1_score": f1_per_class
        })
        self.live.log_plot(
            "f1_by_class",
            df,
            x="note",
            y="f1_score",
            template="bar_horizontal",
            title="F1 score by note"
        )

    def log_average_fp_count_by_silmultaneous_notes_count(self):
        max_silmultaneous = int(np.max(self.y_true.sum(axis=1)))
        avg_fps = []
        for s in range(0, max_silmultaneous + 1):
            indices = self.y_true.sum(axis=1) == s
            avg_fps.append(((self.y_true[indices] == False) & (self.y_pred[indices] >= 0.5)).sum(axis=1).mean())
        df = pd.DataFrame({
            "silmultaneous_notes": list(range(0, max_silmultaneous + 1)),
            "avg_fp": avg_fps
        })

        self.live.log_plot(
            "avg_fp_count_by_silmultaneous_notes_count",
            df,
            x="silmultaneous_notes",
            y="avg_fp",
            template="bar_horizontal",
            title="Avg. false positives count by silmultaneous notes count"
        )

    def log_fp_confusion_matrix_plot(self):
        # For each note we calculate the percentage of false positives of other notes
        conf_matrix = []
        for n in self.notes_vocab.values():
            indices = self.y_true[:, n] == True
            false_positives = ((self.y_pred[indices] >= 0.5) & (self.y_true[indices] == False)).sum(axis=0) / self.y_true.shape[0]
            conf_matrix.append(false_positives)
        conf_matrix = np.array(conf_matrix)

        mock_y_true = []
        mock_y_pred = []
        precision = 3
        for tn, tk in self.notes_vocab.items():
            for pn, pk in self.notes_vocab.items():
                count = int(conf_matrix[tk][pk] * 10 ** precision)
                mock_y_true += count * [tn]
                mock_y_pred += count * [pn]
        
        self.live.log_sklearn_plot("confusion_matrix", mock_y_true, mock_y_pred)

def eval(cfg: Config, live: Live) -> None:
    config = to_config_object(cfg)
    ds_infos = get_datasets_info(config)

    datasets: dict[DsName, tf.data.Dataset] = {}
    for ds_info in ds_infos:
        if ds_info.name in ["val", "test"]:
            datasets[ds_info.name] = create_tf_record_ds(ds_info.config, ds_info.name, shuffle=False)

    model = keras.models.load_model(MODEL_PATH)

    y_true = np.concatenate([y for x, y in datasets["test"]], axis=0)
    y_pred = model.predict(datasets["test"])
    y_pred = tf.sigmoid(y_pred)

    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.numpy().reshape(-1, y_pred.shape[-1])

    print("y_true shape", y_true.shape)
    print("y_pred shape", y_pred.shape)

    _, notes_vocab = load_vocabs(config)

    logger = PlotsLogger(y_true, y_pred, live, notes_vocab)

    logger.log_f1_by_class_plot()
    logger.log_average_fp_count_by_silmultaneous_notes_count()
    logger.log_fp_confusion_matrix_plot()
    
    # TODO: Establish what kind of metrics I want to have here:
    # - bar plot of precision/f1 grouped by individual notes
    # - bar plot of precision/f1 grouped by numer of silmulataneous notes
    # - bar plot of precision/f1 grouped by note_length bins
    # - bar plot of precision/f1 grouped by time_playing / time_remaining bins
    # - If midi is a source: bar plot of precision/f1 grouped by velocity bins (like 30-40, 40-50, 50-60 etc.) 
    # - visualization of some fragment of predictions vs true values (something like notebooks/utils:y_vs_y_pred_vis)
import tensorflow as tf
from dvclive import Live
from musicnet.utils import recreate_dirs
from musicnet.config import Config
from musicnet.config.model import CNNConfig, WaveNetConfig, TransformerConfig
from musicnet.preprocessing.wav_chunks_tfrecord.utils import create_tf_record_ds
from musicnet.models.cnn.train import train as train_cnn
from musicnet.models.transformer.train import train as train_transformer
from musicnet.models.wavenet.train import train as train_wavenet
from musicnet.preprocessing.utils import get_datasets_info
from musicnet.preprocessing.dataset.base import DsName
from musicnet.PipelineState import PipelineState, StageState
from .utils import MODEL_PATH, CHECKPOINT_DIR

def train(config: Config, ps: PipelineState, live: Live) -> None:
    if len(tf.config.list_physical_devices("GPU")) == 0:
        raise Exception("GPU not found")
    
    if ps.stage_state == StageState.CLEAN:
        recreate_dirs([CHECKPOINT_DIR])
    
    ps.set_stage_state(StageState.IN_PROGRESS)
    ds_infos = get_datasets_info(config)

    datasets: dict[DsName, tf.data.Dataset] = {}
    for ds_info in ds_infos:
        datasets[ds_info.name] = create_tf_record_ds(ds_info.config, ds_info.name)

    if isinstance(config.model, CNNConfig):
        train_cnn(datasets["train"], datasets["val"], config.model, live, MODEL_PATH)
    elif isinstance(config.model, TransformerConfig):
        train_transformer(datasets["train"], datasets["val"], config.model, live, MODEL_PATH)
    elif isinstance(config.model, WaveNetConfig):
        train_wavenet(datasets["train"], datasets["val"], config.model, live, MODEL_PATH)
    else:
        raise Exception("Unknown model type")
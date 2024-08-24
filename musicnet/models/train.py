import hydra
import tensorflow as tf
import typing
import os
from dvclive import Live
from .utils import get_training_artifacts_dir
from omegaconf import OmegaConf
from musicnet.utils import PROJECT_ROOT_DIR
from musicnet.config.Config import Config
from musicnet.config.dataset.DatasetConfig import DsConfig
from musicnet.config.model import CNNConfig, WaveNetConfig, TransformerConfig
from musicnet.preprocessing.wav_chunks_tfrecord.utils import create_tf_record_ds
from musicnet.models.cnn.train import train as train_cnn
from musicnet.models.transformer.train import train as train_transformer
from musicnet.models.wavenet.train import train as train_wavenet
from musicnet.preprocessing.utils import get_datasets_info
from musicnet.preprocessing.dataset.base import DsName

@hydra.main(version_base=None, config_name="config")
def train_main(cfg: Config) -> None:
    config = typing.cast(Config, OmegaConf.to_object(cfg))

    if len(tf.config.list_physical_devices("GPU")) == 0:
        raise Exception("GPU not found")
    
    ds_infos = get_datasets_info(config)

    datasets: dict[DsName, tf.data.Dataset] = {}
    for ds_info in ds_infos:
        datasets[ds_info.name] = create_tf_record_ds(ds_info.config, ds_info.name)

    model_path, live_path = get_training_artifacts_dir()

    with Live(dir=live_path, dvcyaml=os.path.join(PROJECT_ROOT_DIR, "dvc.yaml")) as live:
        conf_dict = typing.cast(dict[str, typing.Any], OmegaConf.to_container(cfg, enum_to_str=True))
        live.log_params(conf_dict)
        # Simplified, string version of dataset/model conf
        live.log_param("dataset_cfg", OmegaConf.to_yaml(config.dataset))
        live.log_param("model_cfg", OmegaConf.to_yaml(config.model))
        if isinstance(config.model, CNNConfig):
            train_cnn(datasets["train"], datasets["val"], config.model, live, model_path)
        elif isinstance(config.model, TransformerConfig):
            train_transformer(datasets["train"], datasets["val"], config.model, live, model_path)
        elif isinstance(config.model, WaveNetConfig):
            train_wavenet(datasets["train"], datasets["val"], config.model, live, model_path)
        else:
            raise Exception("Unknown model type")
        
train_main()
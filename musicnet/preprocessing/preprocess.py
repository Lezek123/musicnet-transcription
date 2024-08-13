import hydra
import typing
from omegaconf import OmegaConf
from musicnet.preprocessing.utils import load_dataset
from musicnet.config.Config import Config
from musicnet.config.dataset.preprocessor import WavChunksTFRecordPreprocessorConfig
from musicnet.preprocessing.wav_chunks_tfrecord.preprocess import preprocess as wav_chunks_preprocess

@hydra.main(version_base=None, config_name="config")
def preprocess(cfg: Config):
    config = typing.cast(Config, OmegaConf.to_object(cfg))
    dataset = load_dataset(config)
    preprocessor_cfg = config.dataset.preprocessor
    wav_source_cfg = config.dataset.wav_source

    if isinstance(preprocessor_cfg, WavChunksTFRecordPreprocessorConfig):
        wav_chunks_preprocess(preprocessor_cfg, dataset, wav_source_cfg.type)
    else:
        raise Exception("Unknown preprocessor")
    
preprocess()
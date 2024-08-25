from musicnet.preprocessing.utils import load_source_dataset, get_datasets_info, load_vocabs
from musicnet.config import to_config_object, Config
from musicnet.config.dataset.preprocessor import WavChunksTFRecordPreprocessorConfig
from musicnet.preprocessing.wav_chunks_tfrecord.preprocess import preprocess as wav_chunks_preprocess

def preprocess(cfg: Config):
    config = to_config_object(cfg)
    ds_infos = get_datasets_info(config)
    instruments_vocab, notes_vocab = load_vocabs(config)

    for ds_info in ds_infos:
        dataset = load_source_dataset(ds_info.config, ds_info.src_name)
        preprocessor_cfg = ds_info.config.preprocessor

        if isinstance(preprocessor_cfg, WavChunksTFRecordPreprocessorConfig):
            wav_chunks_preprocess(
                config=preprocessor_cfg,
                dataset=dataset,
                ds_name=ds_info.name,
                instruments_vocab=instruments_vocab,
                notes_vocab=notes_vocab,
                split=ds_info.config.split
            )
        else:
            raise Exception("Unknown preprocessor")
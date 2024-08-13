from enum import Enum
from dataclasses import dataclass

class PreprocessorType(Enum):
    WAV_CHUNKS_TFRECORDS = "wav_chunks_tfrecords"

@dataclass
class Preprocessor:
    type: PreprocessorType
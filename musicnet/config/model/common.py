from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
  CNN = "cnn"
  WAVENET = "wavenet"
  TRANSFORMER = "transformer"

class Activation(Enum):
    GELU = "gelu"
    RELU = "relu"
    ELU = "elu"
    SELU = "selu"

class LRDerivation(Enum):
   AUTO = "auto"

@dataclass
class Model:
    type: ModelType

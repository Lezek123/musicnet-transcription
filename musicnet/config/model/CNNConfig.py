from .common import Model, ModelType, Activation, LRDerivation
from dataclasses import dataclass
from enum import Enum

class ConvType(Enum):
  DEFAULT = "default"
  SEPARABLE = "separable"

@dataclass
class CNNConfig(Model):
  type: ModelType = ModelType.CNN
  n_layers: int = 5
  n_neurons: int = 2048
  activation: Activation = Activation.GELU
  kernel_size: int = 16
  epochs: int = 80
  lr: float | LRDerivation = LRDerivation.AUTO
  dropout_rate: float = 0.
  conv_type: ConvType = ConvType.SEPARABLE
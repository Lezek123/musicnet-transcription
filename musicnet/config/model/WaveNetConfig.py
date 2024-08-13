from dataclasses import dataclass
from .common import Model, Activation, LRDerivation

@dataclass
class WaveNetLayerDef:
    wavenet_kernel: int
    n_maps: int
    pooling: bool

@dataclass
class WaveNetConfig(Model):
  activation: Activation
  epochs: int
  lr: float | LRDerivation
  layers: list[WaveNetLayerDef]

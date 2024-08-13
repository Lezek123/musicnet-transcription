from dataclasses import dataclass
from .common import Model
from enum import Enum

class Architecture(Enum):
  ENCODER = "encoder"
  ENCODER_DECODER = "encoder_decoder"

@dataclass
class TransformerConfig(Model):
  architecture: Architecture
  d_model: int
  num_layers: int
  num_heads: int
  dff: int
  mha_dropout: float
  input_dropout: float
  epochs: int
  warmup_steps: int
  max_lr: float
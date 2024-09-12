import os
import pickle
from omegaconf import OmegaConf
from musicnet.utils import PROJECT_ROOT_DIR, recreate_dirs
from musicnet.config.Config import Config, Stage, to_config_object
from enum import Enum
from dataclasses import dataclass

class StageState(Enum):
    CLEAN = 0
    IN_PROGRESS = 1

STATE_DIR = os.path.join(PROJECT_ROOT_DIR, "state")

@dataclass
class StoredPipelineState():
    current_stage: Stage
    stage_state: StageState
    config: Config

class PipelineState():
    state_file_path = os.path.join(STATE_DIR, "pipeline.pkl")
    current_stage: Stage
    stage_state: StageState

    def __init__(self, config: Config):
        self.config = config
        self.init_state()

    def init_state(self):
        if not os.path.exists(STATE_DIR):
            recreate_dirs([STATE_DIR])
        if self.config.resume == False:
            self.set_stage(Stage.GENERATE)
        else:
            stored_state = self.load_state()
            if stored_state:
                self.current_stage = stored_state.current_stage
                self.stage_state = stored_state.stage_state
                if OmegaConf.to_yaml(stored_state.config.dataset) != OmegaConf.to_yaml(self.config.dataset):
                    print("Dataset parameters changed: resetting the pipeline state")
                    self.set_stage(Stage.GENERATE)
                elif (
                    (OmegaConf.to_yaml(stored_state.config.model) != OmegaConf.to_yaml(self.config.model)) and
                    (self.current_stage.value >= Stage.TRAIN.value)
                ):
                    print("Model parameters changed: moving back to TRAIN stage")
                    self.set_stage(Stage.TRAIN)
            
            # Save the current config and stage data
            self.save_state()

    def load_state(self):
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, "rb") as state_file:
                stored_state: StoredPipelineState = pickle.load(state_file)
            return stored_state
        return None

    def save_state(self):
        with open(self.state_file_path, "wb") as state_file:
            pickle.dump(
                StoredPipelineState(self.current_stage, self.stage_state, self.config),
                state_file
            )

    def set_stage(self, stage: Stage):
        self.current_stage = stage
        self.stage_state = StageState.CLEAN
        self.save_state()

    def set_stage_state(self, stage_state: StageState):
        self.stage_state = stage_state
        self.save_state()
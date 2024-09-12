import hydra
import os
import typing
from omegaconf import OmegaConf
from musicnet.utils import PROJECT_ROOT_DIR
from musicnet.config.Config import Config, Stage, to_config_object
from musicnet.preprocessing.generate import generate
from musicnet.preprocessing.preprocess import preprocess
from musicnet.models.train import train
from dvclive import Live
from musicnet.models.eval import eval
from musicnet.PipelineState import PipelineState, StageState

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    config = to_config_object(cfg)
    ps = PipelineState(config)

    print(f"Starting pipeline from state: {ps.current_stage.name} ({ps.stage_state.name})")

    resume_dvclive = (
        (ps.current_stage == Stage.TRAIN and ps.stage_state == StageState.IN_PROGRESS) or
        (ps.current_stage.value > Stage.TRAIN.value)
    )

    if resume_dvclive:
        print("Resuming from previous point in DVCLive...")

    with Live(
        dir=os.path.join(PROJECT_ROOT_DIR, "dvclive" if cfg.exp else "tmp"),
        save_dvc_exp=cfg.exp,
        dvcyaml=os.path.join(PROJECT_ROOT_DIR, "dvc.yaml") if cfg.exp else None,
        resume=resume_dvclive
    ) as live:
        if not resume_dvclive:
            # Log config parameters with DVCLive
            conf_dict = typing.cast(dict[str, typing.Any], OmegaConf.to_container(cfg, enum_to_str=True))
            live.log_params(conf_dict)
            # Simplified, string version of dataset/model conf
            live.log_param("dataset_cfg", OmegaConf.to_yaml(cfg.dataset))
            live.log_param("model_cfg", OmegaConf.to_yaml(cfg.model))
        
        if ps.current_stage == Stage.GENERATE:
            generate(config, ps=ps)
            ps.set_stage(Stage.PREPROCESS)
        if ps.current_stage == Stage.PREPROCESS:
            preprocess(config, ps=ps)
            ps.set_stage(Stage.TRAIN)
        if ps.current_stage == Stage.TRAIN:
            train(config, ps=ps, live=live)
            ps.set_stage(Stage.EVAL)
        if ps.current_stage == Stage.EVAL:
            eval(config, live=live)

main()
import hydra
import os
from musicnet.utils import PROJECT_ROOT_DIR
from musicnet.config.Config import Config, Stage
from musicnet.preprocessing.generate import generate
from musicnet.preprocessing.preprocess import preprocess
from musicnet.models.train import train
from dvclive import Live
from musicnet.models.eval import eval

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    with Live(
        dir=os.path.join(PROJECT_ROOT_DIR, "dvclive" if cfg.exp else "tmp"),
        save_dvc_exp=cfg.exp,
        dvcyaml=os.path.join(PROJECT_ROOT_DIR, "dvc.yaml") if cfg.exp else None
    ) as live:
        if Stage.GENERATE in cfg.stages:
            generate(cfg)
        if Stage.PREPROCESS in cfg.stages:
            preprocess(cfg)
        if Stage.TRAIN in cfg.stages:
            train(cfg, live)
        if Stage.EVAL in cfg.stages:
            eval(cfg, live)

main()
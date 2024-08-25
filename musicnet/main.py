import hydra
from musicnet.config.Config import Config, Stage
from musicnet.preprocessing.generate import generate
from musicnet.preprocessing.preprocess import preprocess
from musicnet.models.train import train
# from musicnet.models.eval import eval

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    if Stage.GENERATE in cfg.stages:
        generate(cfg)
    if Stage.PREPROCESS in cfg.stages:
        preprocess(cfg)
    if Stage.TRAIN in cfg.stages:
        train(cfg)
    # if Stage.EVAL in cfg.stages:
    #     eval(cfg)

main()
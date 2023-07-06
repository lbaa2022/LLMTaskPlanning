import hydra
from hydra.utils import instantiate

from src.alfred.eval_alfred import AlfredEvaluator


class Evaluator:
    def __init__(self, cfg):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


@hydra.main(version_base=None, config_path="../conf", config_name="config_wah")
def main(cfg):
    evaluator = instantiate(cfg)
    evaluator.evaluate()

if __name__ == "__main__":
    main()

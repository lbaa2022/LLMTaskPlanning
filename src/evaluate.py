import hydra
from hydra.utils import instantiate

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from alfred.eval_alfred import AlfredEvaluator


class Evaluator:
    def __init__(self, cfg):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    print(cfg)
    evaluator = instantiate(cfg)
    evaluator.evaluate()

if __name__ == "__main__":
    main()

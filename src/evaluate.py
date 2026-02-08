import random
import numpy as np
import hydra
from hydra.utils import instantiate

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from src.alfred.alfred_evaluator import AlfredEvaluator
from src.alfred.gt_evaluator import GroundTruthEvaluator
from wah.wah_evaluator import WahEvaluator

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    print(cfg)

    # set random seed
    random.seed(cfg.planner.random_seed)
    np.random.seed(cfg.planner.random_seed)

    if cfg.name == 'alfred':
        evaluator = AlfredEvaluator(cfg)
    elif cfg.name == 'alfred_gt':
        evaluator = GroundTruthEvaluator(cfg)
    elif cfg.name == 'wah':
        evaluator = WahEvaluator(cfg)
    else:
        assert False, f"Unknown evaluator name: {cfg.name}"
    evaluator.evaluate()


if __name__ == "__main__":
    main()

from functools import partial
import logging
from pathlib import Path

import hydra
import wandb

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3 as stb3
from torch.utils.tensorboard.writer import SummaryWriter
from rec_callbacks import REC_CALLBACKS
from tracker import ProgressTracker
from stable_baselines3 import DQN
from extended_dqn import EDQN, DDQN
from utils import set_seed
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger(__name__)

def create_model(cfg, env) -> BaseAlgorithm:
    model_cls_map = {"DQN": DQN, "EDQN": EDQN, "DDQN": DDQN}
    model_cls = model_cls_map[cfg.model.alg]
    policy_kwargs = {
        "net_arch": [cfg.net.width] * cfg.net.depth,
    }
    return model_cls(env=env, policy_kwargs=policy_kwargs, **cfg.model.model)


def create_env(cfg, rec_callbacks=None, writer=None, eval_env=False, monitor=True):
    env = instantiate(cfg.env)
    env.seed(cfg.seed)
    env = Monitor(env)
     
    env = ProgressTracker(
                env=env,
                writer=writer,
                rec_callback=rec_callbacks,
                logpath=Path.cwd() / "images",
                **cfg.tracker,
        )
    return env



@hydra.main(config_path="config", config_name="stable_train")
def main(cfg):
    set_random_seed(cfg.seed)
    set_seed(cfg.seed)
    cfg.model.device = "cpu"
    # import pdb; pdb.set_trace()
    run_name = "{}_seed_{}".format(cfg.model.alg, cfg.seed)
    if "wandb" in cfg:
        wandb.init(
            project="decoupling_representation",
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
            tags=cfg.wandb.tags,
            monitor_gym=True,
            job_type= "{}".format(cfg.model.alg)
        )

    logpath = Path.cwd()
    logger.info(f"Experiment path: {logpath}")
    stb3_logger = configure(str(logpath / "tensorboard"), ["tensorboard", "csv"])
    writer: SummaryWriter = stb3_logger.output_formats[
        0
    ].writer  # Extract the tensorboard logger.

    #import pdb; pdb.set_trace()
    alg = cfg.model
    
    env = create_env(cfg, REC_CALLBACKS[alg], writer)
    model = create_model(cfg, env)
    #import pdb; pdb.set_trace()
    model.learn(eval_env=env, **cfg.learn)



if __name__ == "__main__":
    main()

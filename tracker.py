""" Here we provide a env wrapper that draws some plots."""
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import gym
from gym import Wrapper
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from torch.utils.tensorboard import SummaryWriter    

class ProgressTracker(Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_evals: int,
        num_plots: int,
        plt_list: Sequence[str] = None,
        logpath=None,
        record_video=True,
        writer: SummaryWriter = None,
        prefix: str = None,
        rec_callback: Callable = None,
    ):
        super(ProgressTracker, self).__init__(env)

        plt_list = [] if plt_list is None else list(plt_list)

        # Add plot callbacks of the environment.
        plt_list.append("env")

    
        self.rec_callback = rec_callback

        if record_video:
            assert writer is not None
        self.record_video = record_video

        self.writer = writer
        self.logpath = None if logpath is None else Path(logpath)
        self.prefix = prefix

        if self.logpath is not None:
            self.logpath.mkdir(exist_ok=True)

        self.num_evals = num_evals
        self.num_plots = num_plots
        self.episode_counter = 0
        self.record = {}
    
    def evaluate(self):
        eval_round = self.episode_counter // self.num_evals
        plt_idx = self.episode_counter % self.num_evals



    def reset(self):
        self.episode_reward = 0
        obs = self.env.reset()
        return obs    
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        self.episode_reward += reward
            
        if done:
            self.writer.add_scalar('eval_reward', self.episode_reward, self.episode_counter)
            self.episode_counter += 1
            

        return obs, reward, done, info



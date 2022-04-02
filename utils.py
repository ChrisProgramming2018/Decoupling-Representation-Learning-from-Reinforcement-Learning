import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from typing import Sequence

logger = logging.getLogger(__name__)

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_targe
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    return local_model, target_model



def create_net(input_size, output_size, hidden_sizes: Sequence = None):
        sequence = list()
        if hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip([input_size] + hidden_sizes[:-1], hidden_sizes)
        ]

        for layer in hidden_layers:
            sequence.extend([layer, torch.nn.ReLU()])
        # add last layer without ReLU, also serves as first layer if we have no hidden layers
        last_size = input_size if hidden_sizes == [] else hidden_sizes[-1]
        sequence.append(torch.nn.Linear(last_size, output_size))
        return torch.nn.Sequential(*sequence)

def contrast(forward_mlp, anchor, positives):
    anchor = forward_mlp(anchor)
    pred = (anchor)
    logits = torch.matmul(pred, positives.T)
    return logits


def extract_samples(
    seq_of_obs: torch.tensor,
    prediction_length_k: int, ):
    """
    each obs is allready a framesack, so we can just go util length - k
    """
    samples = []
    for i in range(seq_of_obs.shape[0] - prediction_length_k):
        anchor = seq_of_obs[i]
        positive = seq_of_obs[i + prediction_length_k]
        samples.append(torch.stack((anchor, positive), dim=0))
    return samples

def create_dataset(env, episodes = 1, seed=0):
    samples = []
    env.seed(0)
    env.action_space.seed(0)
    for i_epiosde in range(episodes):
        obs = env.reset()
        seq_of_obs = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            if done:
                break
            seq_of_obs = torch.cat((seq_of_obs, torch.tensor(np.array(next_obs), dtype=torch.float32).unsqueeze(0),), dim=0)
    return extract_samples(seq_of_obs, 10)


def set_seed(seed: Optional[int]):
    """Setting seed to make runs reproducible.

    Args:
        seed: The seed to set.
    """
    if seed is None:
        return

    logger.info(f"Set global seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def time_format(sec):
    """

    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)

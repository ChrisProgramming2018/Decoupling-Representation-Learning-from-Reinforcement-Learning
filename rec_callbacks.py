from collections import defaultdict
from stable_baselines3.dqn.policies import DQNPolicy
import torch
import torch.nn as nn
import numpy as np
import stable_baselines3 as stb3
from stable_baselines3.common.policies import BasePolicy



from extended_dqn import EDQN


def _qvalues_discrete(obs, policy: DQNPolicy):
    data = {}
    device = policy.device

    with torch.no_grad():
        data["qvalues"] = (
            policy.q_net(torch.tensor(obs, device=device)[None, :]).cpu().numpy()
        )
        data["target_qvalues"] = (
            policy.q_net_target(torch.tensor(obs, device=device)[None, :])
            .cpu().numpy()
        )

    return data


def create_dqn_callback(model: stb3.DQN):
    def callback(obs):
        return {**_qvalues_discrete(obs, model.policy)}

    return callback


def create_eqdn_callback(model: EDQN):
    return create_dqn_callback(model)

REC_CALLBACKS = defaultdict(lambda: lambda model: lambda obs: {})


REC_CALLBACKS.update(
    DQN=create_dqn_callback,
    EDQN=create_eqdn_callback,
)

from collections import deque
from typing import Any, Dict, List, Optional, Type, Tuple
from logging import getLogger
import gym
import numpy as np
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn.dqn import DQN
import torch as th
import torch.nn as nn
import torch.nn.functional as F


logger = getLogger(__name__)

class DuellingNetwork(BasePolicy):
    """
    Duelling Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(DuellingNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        advantage_net = create_mlp(
            self.features_dim, action_dim, self.net_arch, self.activation_fn
        )
        self.advantage_net = nn.Sequential(*advantage_net)
        value_net = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        obs = self.extract_features(obs)
        advantage = self.advantage_net(obs)
        return self.value_net(obs) + advantage - advantage.mean(1, keepdim=True)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DuellingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DuellingNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return DuellingNetwork(**net_args).to(self.device)


register_policy("DuellingMlpPolicy", DuellingDQNPolicy)

class EDQN(DQN):
    def __init__(self, lambda_: float = 5.0, **kw):
        self.lambda_ = lambda_
        self.repeat_action = deque()
        super().__init__(**kw)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Argmax with the online network.
                indices = self.q_net(replay_data.next_observations).argmax(dim=1)
                # Follow greedy policy: use the one with the highest value
                next_q_values = next_q_values[range(batch_size), indices]
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not deterministic and np.random.random() < self.exploration_rate:
            num_actions = np.clip(np.random.poisson(lam=self.lambda_), 1, 100)
            self.repeat_action.extend([[self.action_space.sample()]] * num_actions)

        if self.repeat_action and not deterministic:
            action = self.repeat_action.pop()
            logger.debug(f"Exploration Action: {action}")
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
            if not deterministic:
                logger.debug(f"Greedy Action: {action}")

        return action, state

class DDQN(DQN):
    def __init__(self,**kw):
        super().__init__(**kw)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            with th.no_grad():
                # Compute the next Q-values using the target network
                #import pdb; pdb.set_trace()
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Argmax with the online network.
                indices = self.q_net(replay_data.next_observations).argmax(dim=1)
                # Follow greedy policy: use the one with the highest value
                next_q_values = next_q_values[range(batch_size), indices]
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

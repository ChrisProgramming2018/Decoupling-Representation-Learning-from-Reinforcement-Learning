import gym
import random
import torch
import numpy as np
import argparse
import json
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from utils import mkdir



def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('State shape: ', state_size)
    print('Number of actions: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, config=config)
    agent.qnetwork_local.load_state_dict(torch.load(args.modelpath))
    memory = ReplayBuffer((state_size,), (1,), args.max_steps, args.seed, args.device)
    total_maxsteps = args.max_steps
    eps = 0
    rewards = []
    i_episode = 0
    frame = 0
    while frame <= total_maxsteps:
        state = env.reset()
        score = 0
        start_frame = frame
        start_idx = memory.idx
        while True:
            frame += 1
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            score += reward
            memory.add(state, action, reward, next_state, done, done)
            state = next_state
            if frame >= total_maxsteps:
                print("buffer_size ", frame)
                break
            if done:
                i_episode += 1
                if score <= 240:
                    print("current idx memory", memory.idx)
                    print("score to low dont use exampels")
                    frame = start_frame
                    memory.idx = start_idx
                    print("after reset idx memory", memory.idx)
                rewards.append(score)
                print("Episode {} steps {}  Reward {}".format(i_episode, frame, score))
                break
    mean_rewards = np.mean(rewards)
    print("Episode {} average Reward {}".format(i_episode, score))

    mkdir("","expert_policy")
    print("save memory ...")
    memory.save_memory("expert_policy")
    print("... memory saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="LunarLander-v2", type=str)
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--modelpath', default="", type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    arg = parser.parse_args()
    main(arg)

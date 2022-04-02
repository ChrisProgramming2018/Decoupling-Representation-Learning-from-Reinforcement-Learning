from logging.config import valid_ident
from platform import node
import gym
import torch
import numpy as np
import copy
from stable_baselines3 import DQN
import wandb
from utils  import create_dataset, create_net, contrast, soft_update
import argparse


def compute_valid_loss(valid_dataset, encoder, target_encoder, forward_mlp, ce_loss, config, device):
    """ compute  """
    indices = np.arange(len(valid_dataset))
    choices = np.random.choice(indices, size=config.batch_size)
    batch_elements = [valid_dataset[choice] for choice in choices]
    batch = torch.stack(batch_elements).to(device)
    # batch has size [batchsize x 2 x framestack x obs_shape]
    anchors = batch[:, 0]
    positives = batch[:, 1]
    with torch.no_grad():
        c_anchors = encoder(anchors)   # [batchsize x latent_size
        c_positives = target_encoder(positives)        
        logits = contrast(forward_mlp=forward_mlp, anchor=c_anchors, positives=c_positives)
        # 6. get labels for ce loss, as we want 1 vs all, this is just i for the i'th sample
        labels = torch.arange(c_anchors.shape[0], dtype=torch.long, device=device)
    valid_loss = ce_loss(logits, labels)
    return valid_loss.item()

def main(args):
    hyperparameter_defaults = dict(
    nodes_per_layer = 64,
    layers = 2,
    batch_size = 512,
    learning_rate = 1e-4,
    tau=1e-3
    )    
    if args.log:
        wandb.init(
            project="decoupling_representation",
            sync_tensorboard=True,
            save_code=True,
            config=hyperparameter_defaults,
        )
    config = wandb.config
    wandb.run.name = "Nodes {} layers {} bs {} lr {} tau {} ".format(config.nodes_per_layer, config.layers, config.batch_size, config.learning_rate, config.tau)
    env = gym.make("LunarLander-v2")
    policy_kwargs = {
             "net_arch": [config.nodes_per_layer] * config.layers,
             }
    agent = DQN(env=env,policy_kwargs=policy_kwargs, policy = 'MlpPolicy', device='cpu')
    feature_extractor = agent.q_net.features_extractor
    latent_size = agent.q_net.features_dim 

    device= "cpu"
    dataset = create_dataset(env=env, seed=0)
    valid_dataset = create_dataset(env=env, seed=1)
    encoder = copy.deepcopy(feature_extractor).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    forward_mlp = create_net(input_size=latent_size, output_size=latent_size).to(device)
    W = torch.nn.Linear(latent_size, latent_size, bias=False).to(device)
    params = []
    print("encoder params")
    for param in encoder.parameters():
        print(type(param), param.size())
    print("forward_mlp paramas")
    for param in forward_mlp.parameters():
        print(type(param), param.size())
    print("W params")
    for param in W.parameters():
        print(type(param), param.size())

        
    params += encoder.parameters()
    params += forward_mlp.parameters()
    params += W.parameters()
    optim = torch.optim.Adam(params=params,lr=config.learning_rate)
    
    grad_steps = 1000000
    info_freq  = 2500
    print("lat", latent_size)
    
    ce_loss = torch.nn.CrossEntropyLoss()
    for step in range(grad_steps):
        # 1. get batch
        indices = np.arange(len(dataset))
        choices = np.random.choice(indices, size=config.batch_size)
        batch_elements = [dataset[choice] for choice in choices]
        batch = torch.stack(batch_elements).to(device)
        # batch has size [batchsize x 2 x framestack x obs_shape]
        anchors = batch[:, 0]
        positives = batch[:, 1]
        # anchors and positives have size [batchsize x framestack x obs_shape], ready for encoders
        # 2. get anchor_repres with net
        c_anchors = encoder(anchors)  # [batchsize x latent_size]
        # 3. get positives_repres with target_net
        with torch.no_grad():
            c_positives = target_encoder(positives)
        
        logits = contrast(forward_mlp=forward_mlp, anchor=c_anchors, positives=c_positives)
        # 6. get labels for ce loss, as we want 1 vs all, this is just i for the i'th sample
        labels = torch.arange(c_anchors.shape[0], dtype=torch.long, device=device)
        loss = ce_loss(logits, labels)
        optim.zero_grad()
        loss.backward()
        if step % info_freq == 0:
            valid_loss = compute_valid_loss(valid_dataset=valid_dataset, encoder=encoder, target_encoder=target_encoder,forward_mlp=forward_mlp, ce_loss=ce_loss, config=config, device=device)
            wandb.log({'loss': loss.item(), "validation_loss" : valid_loss}, step=step)
            print("Epiosde {} loss {} valid loss {}".format(step, loss.item(), valid_loss))
        # we have grads for W, encoder and forward_mdp but not for target_encoder
        optim.step()
        encoder, target_encoder = soft_update(encoder, target_model=target_encoder, tau=config.tau)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--log', default=True, type=bool) 
    arg = parser.parse_args()
    main(arg)

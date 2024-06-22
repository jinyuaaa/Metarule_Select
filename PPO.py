import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import random as rd
import pandas as pd
import torch
from model import PPO
import config
from utils.env import Mario_env


if __name__ == '__main__':

    # =================== setup ===============================
    parser = argparse.ArgumentParser(description='Reinforcement Meta rule Learner')
    # =================== basic setup =========================
    parser.add_argument('--dataset', type=str,
                        help='dsprites or mario or mnist', default='mario')
    parser.add_argument('--GPU', type=str,
                        help='# of GPU to use', default='1')
    parser.add_argument('--num_cpu_core', type=int,
                        help='CPU core to use', default='16')
    parser.add_argument('--seed', type=int,
                        help='random seed to use in the experiments', default=42)
    args = parser.parse_args()
    args = {**vars(args)}
    config_object = config.config
    config_object['dataset'] = args['dataset']
    config_object['seed'] = args['seed']
    rd.seed(config_object['seed'])
    np.random.seed(config_object['seed'])

    # ==================== parameter setup ====================
    num_episodes = config_object['num_episodes']
    actor_lr = config_object['actor_lr']
    eps = config_object['eps']
    n_hiddens = config_object['n_hiddens']
    epochs= config_object['epochs']
    num_trails_per_episode = config_object['num_trails_per_episode']


    # ==================== path setup =========================
    model_path = config_object['model_path']
    pl_path = config_object['pl_path']
    result_path = config_object['result_path']
    excel_file_path = config_object['excel_result_path']
    log_file_path = config_object['log_file_path']


    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # ==================== device setup =======================
    config_object['GPU'] = args['GPU']
    device = torch.device("cuda:%s" % config_object['GPU'] if torch.cuda.is_available() else "cpu")
    config_object['device'] = device

    # ==================== env setup =========================
    if config_object['dataset'] == 'mario':
        env = Mario_env(config_object)

    # ==================== build model ========================
    state_shape = env.get_state_shape()
    agent = PPO(state_shape, config_object)

    # ==================== on policy training =================
    reward_list = []

    for i in range(num_episodes):
        state = env.reset()
        episode_reward = []

        # save data in this episode
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
        }

        for j in range(num_trails_per_episode):
            action = agent.take_action(np.expand_dims(state,axis=0))
            reward = env.step(action)


            # save states\actions...every timestep
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)

            # new state
            state = env.reset()
            # reward sum
            episode_reward.append(np.mean(reward))
        # save the reward of this episode
        avg_reward = np.mean(episode_reward)
        reward_list.append(avg_reward)
        # train
        transition_dict['states'] = np.array(transition_dict['states'])
        transition_dict['actions'] = np.array(transition_dict['actions'])
        transition_dict['rewards'] = np.array(transition_dict['rewards'])

        agent.learn(transition_dict)

        with open(log_file_path, 'a') as f:
            print(f'iter:{i}, return:{np.mean(reward_list[-1:])}',file=f)
            print(f'iter:{i}, return:{np.mean(reward_list[-1:])}')

        # save model and plot
        if i % 25 == 0:
            torch.save(agent.actor, model_path + 'actor_iter'+str(i)+'.pt')

            plt.plot(reward_list)
            plt.title('reward')
            plt.savefig('./result/result_iter'+str(i)+'.png', bbox_inches='tight')

            data = {
                'reward': reward_list,
            }
            df = pd.DataFrame(data)
            df.to_excel(excel_file_path, index=False)

    # ==================== plot ===========================

    plt.plot(reward_list)
    plt.title('reward')
    plt.savefig('./result/result.png', bbox_inches='tight')

    # ==================== save model ======================
    torch.save(agent.actor, model_path + 'actor_final.pt')

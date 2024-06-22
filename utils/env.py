import numpy as np
import asyncio

from utils.dataloader import MarioDataLoader
from utils.pl_helper import task_writer, bk_writer, run_pl, read_pl_out

class Mario_env:
    def __init__(self, config):
        self.loader = MarioDataLoader(config)
        self.max_pos = config['max_pos']
        self.max_neg = config['max_neg']
        self.num_metarule = config['num_metarule']
        self.max_pic_num = config['max_pic_num']
        self.error_reward = config['error_reward']
        self.B_len = config['B_len']
        self.pl_path = config['pl_path']
        self.num_action_sample = config['num_action_sample']
        self.log_file_path = config['log_file_path']

        # +1:frame type is not designed in B
        self.state_shape = (self.max_pos+self.max_neg, self.max_pic_num+self.B_len+1)
        self.action_shape = (self.num_metarule,)
        self.state = None
        self.task_name = None


    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    # sample a task
    def reset(self):
        # labels_pos:[pos_num_per_bag, label_size]
        # labels_pos:[neg_num_per_bag, label_size]
        self.task_name, labels_pos, labels_neg = self.loader.get_train_data()
        task_writer(self.task_name, labels_pos, labels_neg)
        # state include pos_case, neg_case and Metarule selected status
        # state:[pos_num_per_bag+neg_num_per_bag,size]
        labels_pos_padding = -np.ones((self.max_pos - labels_pos.shape[0],labels_pos.shape[1]))
        labels_neg_padding = -np.ones((self.max_neg - labels_neg.shape[0],labels_pos.shape[1]))
        self.state = np.concatenate((labels_pos, labels_pos_padding, labels_neg, labels_neg_padding), axis=0)
        return self.state

    def step(self, action):
        reward_of_sample_actions = []
        for i in range(self.num_action_sample):
            bk_writer(self.task_name, action[i])
            file_run = self.pl_path + 'task_' + self.task_name + '.pl'

            loop = asyncio.get_event_loop()
            task = loop.create_task(run_pl(file_run))
            loop.run_until_complete(task)

            pl_result, pl_out = task.result()
            with open(self.log_file_path, 'a') as f:
                if pl_result == -1:
                    reward = self.error_reward
                    result = self.task_name + ' sample No.' + str(i) + ':\nERROR\n'
                    result = result + 'reward: ' + str(reward)+'\n'
                    print(result,file=f)
                    print(result)
                elif pl_result == -2:
                    reward = self.error_reward
                    result = self.task_name + ' sample No.' + str(i) + ':\nTime out\n'
                    result = result + 'reward: ' + str(reward)+'\n'
                    print(result,file=f)
                    print(result)
                elif pl_result == 0:
                    reward = -int(np.sum(action[i]))
                    result = self.task_name + ' sample No.' + str(i) + ':\n'
                    result = result + read_pl_out(pl_out)+'\n'
                    result = result + 'reward: '+str(reward)+'\n'
                    print(result,file=f)
                    print(result)
            reward_of_sample_actions.append(reward)
            reward_np = np.array(reward_of_sample_actions)
        return reward_np

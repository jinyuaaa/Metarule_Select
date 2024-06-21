import numpy as np
import random as rd

class MarioDataLoader:
    def __init__(self, config):
        self.dataset = config['dataset']
        dataset_path = config['data_path'] + '/dataset/' +self.dataset +'.npz'
        self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
        # self.imgs_pos = self.data_zip['images_pos'].item()
        # self.imgs_neg = self.data_zip['images_neg'].item()
        self.lbls_pos = self.data_zip['labels_pos'].item()
        self.lbls_neg = self.data_zip['labels_neg'].item()
        self.train_task = config['train_task']
        self.test_task = config['test_task']
        self.train_task_num = len(self.train_task)
        self.test_task_num = len(self.test_task)
        self.num_neg_per_task = len(self.lbls_neg[self.train_task[0]])
        self.pos_train_num = {}
        for key in self.lbls_pos:
            self.pos_train_num[key] = int(len(self.lbls_pos[key]) * (1-config['test_p']))

        self.num_bag_per_batch = config['num_bag_per_batch']
        self.num_pos_case_per_bag = config['num_pos_case_per_bag']
        self.num_neg_case_per_bag = config['num_neg_case_per_bag']

    def get_train_batch(self):
        # sample num_bag_per_batch tasks
        batch_task_index = rd.choices(range(self.train_task_num), k=self.num_bag_per_batch)
        batch_task = [self.train_task[i] for i in batch_task_index]

        # sample positive and negative cases for every tasks
        for i in range(self.num_bag_per_batch):
            this_task_name = batch_task[i]
            pos_train_num_this_task = self.pos_train_num[this_task_name]
            pos_case_index = np.random.choice(range(pos_train_num_this_task), self.num_pos_case_per_bag, replace=False)
            neg_case_index = np.random.choice(range(self.num_neg_per_task), self.num_neg_case_per_bag, replace=False)
            labels_pos_this_task = [self.lbls_pos[this_task_name][i] for i in pos_case_index]
            labels_neg_this_task = [self.lbls_neg[this_task_name][i] for i in neg_case_index]
            # images_neg_this_task = [self.imgs_neg[this_task_name][i] for i in neg_case_index]
            # images_pos_this_task = [self.imgs_pos[this_task_name][i] for i in pos_case_index]
            return labels_pos_this_task, labels_neg_this_task
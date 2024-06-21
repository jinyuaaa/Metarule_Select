import numpy as np
import random as rd

class MarioDataLoader:
    def __init__(self, config):
        self.dataset = config['dataset']
        dataset_path = config['data_path'] + 'dataset/' +self.dataset +'.npz'
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

        self.num_pos = config['num_pos']
        self.num_neg = config['num_neg']
        self.max_pic_num = config['max_pic_num']
        self.grd_map = config['grd_map']
        self.len_guarantee_task =  config['len_guarantee_task']

    def grd_label_to_array_label(self, case):
        array_label = np.zeros((self.max_pic_num + 4), dtype=np.int16)
        for i in range(self.max_pic_num):
            if i < len(case[0]):
                array_label[i] = self.grd_map[case[0][i]]
            else:
                array_label[i] = -1
        array_label[self.max_pic_num] = self.grd_map[case[1][0]]
        for i in range(1, 4):
            array_label[self.max_pic_num + i] = case[1][i]
        return array_label

    def get_train_data(self):
        # sample task
        task_name = rd.choices(self.train_task, k=1)[0]
        labels_pos, labels_neg = self.get_train_data_this_task(task_name)
        return  task_name, labels_pos, labels_neg

    def get_train_data_this_task(self, task_name):
        # sample positive and negative cases for this task
        pos_train_num = self.pos_train_num[task_name]
        n_pos = np.random.choice(self.num_pos, 1)
        n_neg = np.random.choice(self.num_neg, 1)
        pos_case_index = np.random.choice(range(pos_train_num), n_pos, replace=False)
        neg_case_index = np.random.choice(range(self.num_neg_per_task), n_neg, replace=False)

        # priority and jump task should guarantee at least one longest case is chosen
        if task_name in self.len_guarantee_task:
            flag = True
            while flag:
                for index in pos_case_index:
                    if len(self.lbls_pos[task_name][index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag:
                    pos_case_index = np.random.choice(range(pos_train_num), n_pos, replace=False)


        labels_pos = [self.grd_label_to_array_label(self.lbls_pos[task_name][i]) for i in pos_case_index]
        labels_neg = [self.grd_label_to_array_label(self.lbls_neg[task_name][i]) for i in neg_case_index]
        # images_neg_this_task = [self.imgs_neg[this_task_name][i] for i in neg_case_index]
        # images_pos_this_task = [self.imgs_pos[this_task_name][i] for i in pos_case_index]

        labels_pos = np.array(labels_pos)
        labels_neg = np.array(labels_neg)
        return labels_pos, labels_neg
import numpy as np
import torch
from torch import nn

class PolicyNet(nn.Module):
    def __init__(self, state_shape, config):
        super(PolicyNet, self).__init__()
        self.device = config['device']
        self.metarule_encoding_map = config['metarule_encoding_map']
        self.n_hiddens = config['n_hiddens']
        self.num_metarule = config['num_metarule']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.max_pos = config['max_pos']
        self.max_neg = config['max_neg']
        self.log_file_path = config['log_file_path']

        # [num_meta_rule, encoding_size]
        self.metarule_encoding = torch.tensor([self.metarule_encoding_map[i] for i in range(self.num_metarule)], dtype=torch.float32).to(self.device)

        self.fc_pos = nn.Linear(state_shape[1], self.n_hiddens)
        self.fc_neg = nn.Linear(state_shape[1], self.n_hiddens)
        self.SelfAttention_pos = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.SelfAttention_neg = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.SelfAttention_case = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.fc_metarule = nn.Linear(self.metarule_encoding.shape[1], self.n_hiddens)
        self.CrossAttention_case_metarule = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.fc_prob = nn.Linear(self.n_hiddens, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [batch, max_pos, encoding_dim]
        pos_encoding = x[:,0:self.max_pos,:].clone().float().to(self.device)
        # [batch, max_neg, encoding_dim]
        neg_encoding = x[:,self.max_pos:,:].clone().float().to(self.device)

        # [batch, max_pos, hidden]
        pos_hidden = self.fc_pos(pos_encoding)
        pos_Attention, _= self.SelfAttention_pos(pos_hidden,pos_hidden,pos_hidden)
        # [batch, max_neg, hidden]
        neg_hidden = self.fc_neg(neg_encoding)
        neg_Attention, _ = self.SelfAttention_neg(neg_hidden,neg_hidden,neg_hidden)

        # [batch, max_pos+max_neg, n_hiddens]
        case_hidden = torch.concat((pos_Attention, neg_Attention), dim=1)
        case_Attention, _ = self.SelfAttention_case(case_hidden, case_hidden, case_hidden)
        # [batch, num_metarule, n_hiddens]
        metarule_hidden = self.fc_metarule(torch.repeat_interleave(self.metarule_encoding.unsqueeze(0),x.shape[0],dim=0))
        # [batch, num_metarule, n_hiddens]
        metarule_CrossAttention, _ = self.CrossAttention_case_metarule(metarule_hidden, case_Attention, case_Attention)
        # [batch, num_metarule, 1]
        metarule_prob = torch.clamp(self.Sigmoid(self.fc_prob(metarule_CrossAttention)), min=1e-7, max=1 - 1e-7)
        with open(self.log_file_path, 'a') as f:
            print(metarule_prob, file=f)
            print(metarule_prob)
        return metarule_prob

class PPO:
    def __init__(self, state_shape, config, train=True):
        self.device = config['device']
        self.train = train

        self.actor = PolicyNet(state_shape, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])

        self.epochs = config['epochs']
        self.eps = config['eps']

        self.num_metarule = config['num_metarule']
        self.num_action_sample = config['num_action_sample']

    def take_action(self, state):
        # [max_pos + max_neg, encoding_dim]
        state = torch.tensor(state).to(self.device)
        # [num_metarules, 1]
        probs = self.actor(state).view(-1)
        # Create a probability distribution based on probs
        bernoulli_dist = torch.distributions.Bernoulli(probs)
        # Pick metarules based on its probability
        action = bernoulli_dist.sample().unsqueeze(dim=0)
        if self.train:
            for i in range(self.num_action_sample-1):
                action=torch.concat((action,bernoulli_dist.sample().unsqueeze(dim=0)),dim=0)
        action = np.array(action.to('cpu'),dtype=np.int16)
        return action

    def learn(self, transition_dict):
        # get data
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, self.num_action_sample, self.num_metarule, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1,self.num_action_sample, 1)

        # advantage function
        adv_same_task = torch.mean(rewards, dim=1, keepdim=True)
        adv_all = torch.mean(adv_same_task, dim=0, keepdim=True)
        advantage = (rewards - 0.5*adv_all-0.5*adv_same_task).to(self.device)
        # advantage = (rewards - adv_same_task).to(self.device) #TODO: which?

        probs = torch.repeat_interleave(self.actor(states).unsqueeze(dim=1), self.num_action_sample, dim=1)
        prob_action = torch.prod(probs*actions + (1-probs)*(1-actions),dim=2)

        old_log_probs = torch.log(prob_action).detach()

        # A set of data is trained epochs round
        for _ in range(self.epochs):
            probs = torch.repeat_interleave(self.actor(states).unsqueeze(dim=1), self.num_action_sample, dim=1)
            prob_action = torch.prod(probs * actions + (1 - probs) * (1 - actions), dim=2)
            # Update the state of the policy network forecast once a round
            log_probs = torch.log(prob_action)
            # The ratio between old and new probs
            ratio = torch.exp(log_probs - old_log_probs)
            # left
            surr1 = ratio * advantage
            # right
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # Loss function of policy network
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


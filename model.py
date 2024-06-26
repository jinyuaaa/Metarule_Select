import numpy as np
import torch
import torch.optim as optim
from torch import nn

class PolicyNet(nn.Module):
    def __init__(self, state_shape, config):
        super(PolicyNet, self).__init__()
        self.device = config['device']
        # self.metarule_encoding_map = config['metarule_encoding_map']
        self.n_hiddens = config['n_hiddens']
        self.num_metarule = config['num_metarule']
        self.metarule_embedding_dim = config['metarule_embedding_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.max_pos = config['max_pos']
        self.max_neg = config['max_neg']
        self.log_file_path = config['log_file_path']
        self.padding_mask = config['padding_mask']
        self.relu = config['relu']

        # [num_meta_rule, encoding_size]
        # self.metarule_encoding = torch.tensor([self.metarule_encoding_map[i] for i in range(self.num_metarule)], dtype=torch.float32).to(self.device)

        self.embedding = nn.Embedding(self.num_metarule, self.metarule_embedding_dim)      #TODO:how to embed metarule

        if self.relu:
            self.fc_pos = nn.Sequential(
                nn.Linear(state_shape[1],  self.n_hiddens),
                nn.ReLU(),
                nn.Linear( self.n_hiddens,  self.n_hiddens)
            )
            self.fc_neg = nn.Sequential(
                nn.Linear(state_shape[1], self.n_hiddens),
                nn.ReLU(),
                nn.Linear(self.n_hiddens, self.n_hiddens)
            )
            self.fc_metarule = nn.Sequential(
                nn.Linear(self.metarule_embedding_dim, self.n_hiddens),
                nn.ReLU(),
                nn.Linear(self.n_hiddens, self.n_hiddens)
            )
        else:
            self.fc_pos = nn.Linear(state_shape[1],  self.n_hiddens)
            self.fc_neg = nn.Linear(state_shape[1], self.n_hiddens)
            self.fc_metarule = nn.Linear(self.metarule_embedding_dim, self.n_hiddens)

        self.SelfAttention_pos = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.SelfAttention_neg = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.SelfAttention_case = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)



        # self.fc_metarule = nn.Sequential(
        #     nn.Linear(self.metarule_encoding.shape[1],  self.n_hiddens),
        #     # nn.ReLU(),
        #     # nn.Linear( self.n_hiddens,  self.n_hiddens)
        # )

        self.CrossAttention_case_metarule = nn.MultiheadAttention(embed_dim=self.n_hiddens, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

        self.fc_prob = nn.Sequential(
            nn.Linear(self.n_hiddens,  1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [batch, max_pos, encoding_dim]
        pos_encoding = x[:,0:self.max_pos,:].clone().float().to(self.device)
        # [batch, max_neg, encoding_dim]
        neg_encoding = x[:,self.max_pos:,:].clone().float().to(self.device)

        # [batch, max_pos, hidden]
        pos_hidden = self.fc_pos(pos_encoding)
        # [batch, max_neg, hidden]
        neg_hidden = self.fc_neg(neg_encoding)

        if self.padding_mask:
            pos_mask = (pos_encoding == -1).all(dim=2)
            neg_mask = (neg_encoding == -1).all(dim=2)
            pos_Attention, _ = self.SelfAttention_pos(pos_hidden, pos_hidden, pos_hidden, key_padding_mask=pos_mask)
            neg_Attention, _ = self.SelfAttention_neg(neg_hidden, neg_hidden, neg_hidden, key_padding_mask=neg_mask)
        else:
            pos_Attention, _ = self.SelfAttention_pos(pos_hidden, pos_hidden, pos_hidden)
            neg_Attention, _ = self.SelfAttention_neg(neg_hidden, neg_hidden, neg_hidden)

        # [batch, max_pos+max_neg, n_hiddens]
        case_hidden = torch.concat((pos_Attention, neg_Attention), dim=1)
        case_Attention, _ = self.SelfAttention_case(case_hidden, case_hidden, case_hidden)

        # [batch, num_metarule, n_hiddens]
        metarule_hidden = self.fc_metarule(torch.repeat_interleave(self.embedding.weight.unsqueeze(0),x.shape[0],dim=0))
        # metarule_hidden = self.fc_metarule(torch.repeat_interleave(self.metarule_encoding.unsqueeze(0),x.shape[0],dim=0))

        # [batch, num_metarule, n_hiddens]
        metarule_CrossAttention, _ = self.CrossAttention_case_metarule(metarule_hidden, case_Attention, case_Attention)
        # [batch, num_metarule, 1]
        metarule_prob = torch.clamp(self.fc_prob(metarule_CrossAttention), min=1e-7, max=1 - 1e-7)
        with open(self.log_file_path, 'a') as f:
            print(metarule_prob, file=f)
            print(metarule_prob)
        return metarule_prob

class PPO:
    def __init__(self, state_shape, config):
        self.device = config['device']

        self.actor = PolicyNet(state_shape, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        # self.scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

        self.epochs = config['epochs']
        self.eps = config['eps']
        self.epsilon = config['epsilon']
        self.epsilon_decay_rate = config['epsilon_decay_rate']

        self.num_metarule = config['num_metarule']
        self.num_action_sample = config['num_action_sample']

        self.log_file_path = config['log_file_path']

        self.gae_mode = config['gae_mode']

    def take_action(self, state, mode):
        # [batch, max_pos + max_neg, encoding_dim]
        state = torch.tensor(state).to(self.device)
        # [num_metarules, 1]
        probs = self.actor(state).view(-1)
        # Create a probability distribution based on probs
        bernoulli_dist = torch.distributions.Bernoulli(probs)

        action = bernoulli_dist.sample().unsqueeze(dim=0)
        if mode == 'train':
            for i in range(self.num_action_sample-1):
                if np.random.rand() < self.epsilon:
                    # Pick metarule randomly
                    action = torch.concat((action, torch.randint(0, 2, (1, self.num_metarule)).to(self.device)),0)
                else:
                    # Pick metarules based on its probability
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
        if self.gae_mode == 'all':
            advantage = (rewards - 0.5*adv_all-0.5*adv_same_task).to(self.device)
        elif self.gae_mode == 'same_task':
            advantage = (rewards - adv_same_task).to(self.device) #TODO: which?
        else:
            advantage = rewards

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

        # self.scheduler.step()
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        self.epsilon = self.epsilon*self.epsilon_decay_rate

        with open(self.log_file_path, 'a') as f:
            print('lr_now: '+str(current_lr), file=f)
            print('lr_now: '+str(current_lr))
            print('Exploration rate: '+str(self.epsilon), file=f)
            print('Exploration rate: '+str(self.epsilon))



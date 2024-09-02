import numpy as np
import Envi_util as EnviU
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=400, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, 600)
        self.linear2 = nn.Linear(600, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=300, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, 800)
        self.linear2 = nn.Linear(800, 200)
        self.linear3 = nn.Linear(200, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class TD3:
    def __init__(self, request_num, provider_num, actor_lr, critic_lr, sigma, tau, gamma, MEMORY_CAPACITY, BATCH_SIZE):
        self.request_num = request_num
        self.provider_num = provider_num
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.n_states = 2 * self.request_num * self.provider_num + 2 * self.provider_num
        self.n_actions = (self.request_num + 2) * self.provider_num
        self.critic1 = Critic(self.n_states, self.n_actions)
        self.critic2 = Critic(self.n_states, self.n_actions)
        self.actor = Actor(self.n_states, self.n_actions)
        self.target_critic1 = Critic(self.n_states, self.n_actions)
        self.target_critic2 = Critic(self.n_states, self.n_actions)
        self.target_actor = Actor(self.n_states, self.n_actions)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.learn_step_counter = 0
        self.K = 4
        self.memory_counter = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.n_states * 2 + self.n_actions + 1))
        self.loss_func = nn.MSELoss()


    def choose_action(self, data_number, data_quality, uav_user_channel, user_uav_trust):
        uav_user_channel_new = np.zeros(self.provider_num * self.request_num)
        for i in range(self.provider_num):
            uav_user_channel_new[(i * self.request_num):(i * self.request_num + self.request_num)] = uav_user_channel[i, :]
        user_uav_trust_new = np.zeros(self.request_num * self.provider_num)
        for i in range(self.request_num):
            user_uav_trust_new[(i * self.provider_num):(i * self.provider_num + self.provider_num)] = user_uav_trust[i, :]
        state = np.hstack((data_number, data_quality, uav_user_channel_new, user_uav_trust_new))
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.actor.forward(state).data.numpy().squeeze()
        actions_value = actions_value + self.sigma * np.random.randn(self.n_actions)
        request_strategy, power_strategy, computation_strategy = EnviU.get_three_strategy(actions_value, self.request_num, self.provider_num)
        return request_strategy, power_strategy, computation_strategy

    def soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def store_transition(self, data_number, data_quality, uav_user_channel, user_uav_trust, request_strategy, power_strategy, computation_strategy, r, data_number_next, data_quality_next, uav_user_channel_next, user_uav_trust_next):
        uav_user_channel_new = np.zeros(self.provider_num * self.request_num)
        user_uav_trust_new = np.zeros(self.request_num * self.provider_num)
        request_strategy_new = np.zeros(self.request_num * self.provider_num)
        uav_user_channel_next_new = np.zeros(self.provider_num * self.request_num)
        user_uav_trust_next_new = np.zeros(self.request_num * self.provider_num)
        for i in range(self.provider_num):
            uav_user_channel_new[(i * self.request_num):(i * self.request_num + self.request_num)] = uav_user_channel[i, :]
            uav_user_channel_next_new[(i * self.request_num):(i * self.request_num + self.request_num)] = uav_user_channel_next[i, :]
        for i in range(self.request_num):
            request_strategy_new[(i * self.provider_num):(i * self.provider_num + self.provider_num)] = request_strategy[i, :]
            user_uav_trust_new[(i * self.provider_num):(i * self.provider_num + self.provider_num)] = user_uav_trust[i, :]
            user_uav_trust_next_new[(i * self.provider_num):(i * self.provider_num + self.provider_num)] = user_uav_trust_next[i, :]
        transition = np.hstack((data_number, data_quality, uav_user_channel_new, user_uav_trust_new, request_strategy_new, power_strategy, computation_strategy, [r], data_number_next, data_quality_next, uav_user_channel_next_new, user_uav_trust_next_new))
        if self.memory_counter <= (self.MEMORY_CAPACITY - 1):
            index = self.memory_counter % self.MEMORY_CAPACITY
            self.memory[index, :] = transition
        else:
            min_row = np.where(self.memory[:, (self.n_states + self.n_actions)] == np.min(self.memory[:, (self.n_states + self.n_actions)]))
            index = min_row[0][0]
            self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.FloatTensor(b_memory[:, self.n_states:(self.n_states + self.n_actions)])
        b_reward = torch.FloatTensor(b_memory[:, (self.n_states + self.n_actions):(self.n_states + self.n_actions + 1)])
        b_next_state = torch.FloatTensor(b_memory[:, (-self.n_states):])

        next_value1 = self.target_critic1(b_next_state, self.target_actor(b_next_state))
        next_value2 = self.target_critic2(b_next_state, self.target_actor(b_next_state))
        target_value = b_reward + self.gamma * torch.min(next_value1, next_value2)
        critic_loss1 = self.loss_func(self.critic1(b_state, b_action), target_value)
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward(retain_graph=True)
        self.critic_optimizer1.step()

        critic_loss2 = self.loss_func(self.critic2(b_state, b_action), target_value)
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if self.learn_step_counter % self.K == 0:
            actor_loss = -torch.mean(self.critic1(b_state, self.actor(b_state)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update()

request_num = 10
provider_num = 10
wireless_band = 20000
model_size = 2400
train_number = 4
comp_requester = np.random.uniform(low=1.0, high=2.0, size=request_num)
similar_value = np.random.uniform(low=0.75, high=0.95, size=(request_num, provider_num))
number_paras = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
number_trans_prob = np.array([[0.5, 0.15, 0.125, 0.12, 0.105], [0.15, 0.5, 0.125, 0.12, 0.105], [0.105, 0.15, 0.5, 0.125, 0.12], [0.105, 0.12, 0.125, 0.5, 0.15], [0.105, 0.12, 0.125, 0.15, 0.5]])
quality_paras = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
quality_trans_prob = np.array([[0.5, 0.15, 0.125, 0.12, 0.105], [0.15, 0.5, 0.125, 0.12, 0.105], [0.105, 0.15, 0.5, 0.125, 0.12], [0.105, 0.12, 0.125, 0.5, 0.15], [0.105, 0.12, 0.125, 0.15, 0.5]])
channel_paras = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
channel_trans_prob = np.array([[0.5, 0.15, 0.125, 0.12, 0.105], [0.15, 0.5, 0.125, 0.12, 0.105], [0.105, 0.15, 0.5, 0.125, 0.12], [0.105, 0.12, 0.125, 0.5, 0.15], [0.105, 0.12, 0.125, 0.15, 0.5]])
trust_paras = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
trust_trans_prob = np.array([[0.5, 0.15, 0.125, 0.12, 0.105], [0.15, 0.5, 0.125, 0.12, 0.105], [0.105, 0.15, 0.5, 0.125, 0.12], [0.105, 0.12, 0.125, 0.5, 0.15], [0.105, 0.12, 0.125, 0.15, 0.5]])
delay_threshold = 200

power_max = 1
comp_max = 2
actor_lr = 5e-4
critic_lr = 5e-3
sigma = 0.005
tau = 0.005
gamma = 0.95
MEMORY_CAPACITY = 2000
BATCH_SIZE = 64

num_episodes = 400
slot_times = 20

env = EnviU.IntelligenceSharingEnv(request_num, provider_num, wireless_band, model_size, train_number, comp_requester, similar_value, number_paras, number_trans_prob, quality_paras, quality_trans_prob, channel_paras, channel_trans_prob, trust_paras, trust_trans_prob, delay_threshold, power_max, comp_max)

agent = TD3(request_num, provider_num, actor_lr, critic_lr, sigma, tau, gamma, MEMORY_CAPACITY, BATCH_SIZE)

return_list1 = []

for i_episode in range(num_episodes):
    print('i_episode:' + str(i_episode))
    episode_return1 = 0
    state_data_number, state_data_quality, state_channel, state_trust = env.reset()
    for i_slot in range(slot_times):
        reward = float('nan')
        while np.isnan(reward):
            request_strategy, power_strategy, computation_strategy = agent.choose_action(state_data_number.copy(), state_data_quality.copy(), state_channel.copy(), state_trust.copy())
            next_state_data_number, next_state_data_quality, next_state_channel, next_state_trust, reward = env.step(request_strategy.copy(), power_strategy.copy(), computation_strategy.copy())
        agent.store_transition(state_data_number, state_data_quality, state_channel, state_trust, request_strategy, power_strategy, computation_strategy, reward, next_state_data_number, next_state_data_quality, next_state_channel, next_state_trust)
        episode_return1 += reward

        state_data_number = next_state_data_number
        state_data_quality = next_state_data_quality
        state_channel = next_state_channel
        state_trust = next_state_trust
        if agent.memory_counter > MEMORY_CAPACITY:
            agent.learn()
    return_list1.append(episode_return1)

return_list1 = np.array(return_list1)
print("TD3-based IR, PC and CA：", np.mean(return_list1[350:399]))


# -*- coding: gb18030 -*-

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.states_ = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.states_), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, state_, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.states_.append(state_)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states_ = []


class ActorNetwork(nn.Module):
    def __init__(self, action_dim, input_dims, alpha, fc1_dims=128, fc2_dims=128):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(in_features=input_dims, out_features=fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, action_dim)
        )

        self.log_std = nn.Parameter(T.zeros(action_dim))

        self.alpha = alpha
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # mean = T.split(T.tanh(self.actor(state)), 2, dim=-1)[0]
        # std = T.split(abs(T.tanh(self.actor(state))), 2, dim=-1)[1]
        mean = T.tanh(self.actor(state))
        std = T.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=128, fc2_dims=128):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.alpha = alpha
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value


class Agent:
    def __init__(self, action_dim, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(action_dim, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, state_, reward, done):
        self.memory.store_memory(state, action, probs, state_, reward, done)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        action = dist.sample()

        log_probs = dist.log_prob(action)
        log_probs = log_probs.sum(-1, keepdim=True).detach().numpy()
        action = action.detach().numpy()

        return action, log_probs

    def learn(self, alpha):
        global out_actor_loss, out_critic_loss, total_loss, out_entropy
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, state__arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            with T.no_grad():
                dones_arr = dones_arr.astype(int)
                v = self.critic(T.tensor(state_arr)).flatten()
                v_ = self.critic(T.tensor(state__arr)).flatten()
                deltas = T.tensor(reward_arr) + self.gamma * T.mul(v_, (T.ones(len(dones_arr)) - T.tensor(dones_arr))) - v
                deltas = deltas.flatten().numpy().tolist()
                gae = 0
                discount = 1
                advantage = []
                for index in range(1, len(deltas)+1):
                    delta = deltas[-index]
                    discount = 1 if dones_arr[-index] == 1 else discount
                    gae *= (1 - dones_arr[-index]) * discount
                    gae += delta
                    discount *= self.gamma * self.gae_lambda
                    advantage.insert(0, gae)

            # for t in range(len(reward_arr) - 1):
            #     if dones_arr[t] == 1:
            #     discount = 1
            #     a_t = 0
            #     for k in range(t, len(reward_arr) - 1):
            #         a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
            #         discount *= self.gamma * self.gae_lambda
            #     advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(v).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                new_dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = new_dist.log_prob(actions)
                new_probs = new_probs.sum(1, keepdim=True)
                prob_ratio = new_probs.exp() / old_probs.exp()
                dist_entropy = new_dist.entropy().sum(1, keepdim=True)
                surr1 = advantage[batch] * prob_ratio
                surr2 = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = (-T.min(surr1, surr2)).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                entropy_loss = - dist_entropy.mean()

                out_actor_loss = actor_loss
                out_critic_loss = critic_loss
                out_entropy = - entropy_loss

                total_loss = actor_loss + 0.5 * critic_loss + 0.001 * entropy_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # actor_loss.backward()
                # critic_loss.backward()
                self.actor.alpha = alpha
                self.critic.alpha = alpha
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        return out_actor_loss, out_critic_loss, out_entropy, total_loss

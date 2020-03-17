#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import gym
from keras import Sequential, regularizers, optimizers
from keras.layers import Dense
from collections import deque
import random
import pprint as pp

import argparse

class DQN(object):
    def __init__(self, s_dim, a_dim, batch_size, learning_rate):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.lr = learning_rate
        
    def _build_net(self):
        model = Sequential()
        model.add(Dense(120, input_dim=self.s_dim, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
        model.add(Dense(64, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
        model.add(Dense(self.a_dim, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.adam(lr=self.lr))
        return model

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()
        self.count = 0


def learn_from_batch(replay_buffer, dqn, batch_size, gamma, s_dim, a_dim):

    #print('---------- Learn from Batch ----------')
    samples = replay_buffer.sample_batch(batch_size)
    state_batch = np.array([_[0] for _ in samples])        # (32, 8) 
    action_batch = np.array([_[1] for _ in samples])      # (32,1)
    reward_batch = np.array([_[2] for _ in samples])      # (32,)
    next_state_batch = np.array([_[3] for _ in samples])   # (32, 8) 
    done_batch = np.array([_[4] for _ in samples])

    state_batch = np.squeeze(state_batch)
    next_state_batch = np.squeeze(next_state_batch)

    targets = reward_batch + gamma*(np.amax(dqn.predict_on_batch(next_state_batch), axis=1))*(1-done_batch)
    targets_full = dqn.predict_on_batch(state_batch)
    ind = np.array([i for i in range(batch_size)])
    targets_full[[ind], [action_batch]] = targets

    dqn.fit(state_batch, targets_full, epochs=1, verbose=0)
    

def train(env, dqn, s_dim, a_dim, args):    
    # initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']))

    epsilon = float(args['epsilon'])
    
    score = []
    for i in range(int(args['max_episodes'])):
        state = env.reset()     # Initialize state
        ep_reward = 0.0
        
        for j in range(args['max_episodes_len']): 

            epsilon = max(epsilon, 0.01)
            ## Epsilon-Greedy 
            if np.random.rand() > epsilon:
                q_value = dqn.predict(np.reshape(state, [1, s_dim]))
                action = np.argmax(q_value[0])
            else:
                action = np.random.choice([0,1,2,3])
            epsilon *= float(args['epsilon_decay'])

            #print(f"action: {action}")
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
           
            replay_buffer.add(tuple(state.reshape((s_dim,))),
                                action,
                                reward,
                                tuple(next_state.reshape((s_dim,))),
                                done)

            if replay_buffer.size() > int(args['batch_size']):
                learn_from_batch(replay_buffer, dqn, int(args['batch_size']), float(args['gamma']), s_dim, a_dim)

            # move to next state 
            state = next_state
            
            if done:
            	#print(f"Episode finished after {j+1} steps")
            	break

        score.append(ep_reward)
        print(f"========== Episode {i+1}, Total {j+1} Rounds : Reward {ep_reward:.3f}  =========")
        print(f"Average reward of last 100 episodes: {np.mean(score[-100:]):.3f}")


def main(args):
    # simulated environment
    env = gym.make('LunarLander-v2')
    env.seed(100)
    np.random.seed(100)
    # state total dimension  
    s_dim = 8
    # action (total number of choice)
    a_dim = 4
    dqn = DQN(s_dim, a_dim, int(args['batch_size']), float(args['lr']))._build_net()
    train(env, dqn, s_dim, a_dim, args)
    env.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--lr", help="dqn network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--epsilon", help="epsilon-greedy parameter", default=0.95)
    parser.add_argument("--epsilon_decay", help="epsilon-greedy parameter", default=0.995)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)  # default = 64

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=500)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=2000)    # defult = 100
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')

    args_ = vars(parser.parse_args())
    pp.pformat(args_)

    main(args_)


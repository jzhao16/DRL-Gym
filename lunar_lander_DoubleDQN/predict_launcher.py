#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow 
from tensorflow.keras.models import load_model
from collections import deque
import random
import pprint as pp

import argparse

def predict(env, dqn, s_dim, a_dim, args):    
    
    score = []
    for i in range(int(args['max_episodes'])):
        state = env.reset()     # Initialize state
        ep_reward = 0.0
        
        for j in range(args['max_episodes_len']): 
            q_value = dqn.predict(np.reshape(state, [1, s_dim]))
            action = np.argmax(q_value[0])

            env.render()
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
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
    # state total dimension  
    s_dim = 8
    # action (total number of choice)
    a_dim = 4
    # load model
    dqn = load_model('double_dqn.h5')
    predict(env, dqn, s_dim, a_dim, args)
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

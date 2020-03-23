#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by py2pi on 2019/08/08
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random
import gym
import time 
import pprint as pp

from util.logger import logger

import argparse

class Actor(object):
    """policy function approximator"""
    def __init__(self, sess, s_dim, a_dim, batch_size, tau, learning_rate, global_step_tensor = None, scope="actor"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.global_step = global_step_tensor
        self.scope = scope

        with tf.compat.v1.variable_scope(self.scope):
            self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
            # estimator actor network
            self.state, self.action = self._build_net("online_actor")
            self.network_params = tf.compat.v1.trainable_variables()    # len = 4 
            # self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_actor')

            # target actor network
            self.target_state, self.target_action = self._build_net("target_actor")
            self.target_network_params = tf.compat.v1.trainable_variables()[len(self.network_params):]   # len = 4
            # self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assign(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.a_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])    # action gradient, dQ/da, (1, 120)
            
            self.params_gradients = list(                                       
                map(
                    lambda x: tf.math.divide(x, self.batch_size),              
                    tf.gradients(tf.reshape(self.action, [self.batch_size, self.a_dim]),          # tf.gradients(ys, xs, grad_ys=None, name='gradients')
                                 self.network_params, -self.a_gradient)                           # chain_rule
                )
            )
           
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(self.params_gradients, self.network_params), global_step=self.global_step
            )
            self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def _build_net(self, scope):
        """build the tensorflow graph"""
        with tf.compat.v1.variable_scope(scope):
            state = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], "state")                         # (1, 360)
            hidden1 = keras.layers.Dense(200, activation='relu', use_bias=False, 
                                            kernel_regularizer=keras.regularizers.l2(0.01))(state)
            batchNorm1 = keras.layers.BatchNormalization(axis=-1)(hidden1)
            hidden2 = keras.layers.Dense(64, activation='relu', use_bias=False,                         
                                            kernel_regularizer=keras.regularizers.l2(0.01))(batchNorm1)
            batchNorm2 = keras.layers.BatchNormalization(axis=-1)(hidden2)                                 # (1, 360)
            output = keras.layers.Dense(self.a_dim, activation='tanh', use_bias=False)(batchNorm2)
        return state, output

    def train(self, state, a_gradient):
        self.sess.run([self.optimizer, self.params_gradients], feed_dict={self.state: state, self.a_gradient: a_gradient, self.is_training: True})
        #print(params_gradients)

    def predict(self, state):
        return self.sess.run(self.action, feed_dict={self.state: state, self.is_training: True})

    def predict_target(self, state):
        return self.sess.run(self.target_action, feed_dict={self.target_state: state, self.is_training: True})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    """value function approximator"""
    def __init__(self, sess, s_dim, a_dim, num_actor_vars, gamma, tau, learning_rate, scope="critic"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_actor_vars = num_actor_vars
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.compat.v1.variable_scope(self.scope):
            # estimator critic network
            self.state, self.action, self.q_value = self._build_net("online_critic")
            self.network_params = tf.compat.v1.trainable_variables()[self.num_actor_vars:]
            # self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator_critic")

            # target critic network
            self.target_state, self.target_action, self.target_q_value = self._build_net("target_critic")
            self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + self.num_actor_vars):]
            # self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]

            self.hard_update_target_network_params = [
                 self.target_network_params[i].assign(
                     self.network_params[i]
                 ) for i in range(len(self.target_network_params))
             ]
            # predicted Q value from Bellman's equation
            self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])      
            # The loss between calculated Q from Critic vs predicted Q from Bellman's equation (Huber Loss or MSE)
            self.loss = tf.compat.v1.losses.mean_squared_error(self.predicted_q_value, self.q_value)                  
            
            #self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)  # default AdamOptimizer
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.a_gradient = tf.gradients(self.q_value, self.action)    # dQ(critic) / da 

    def _build_net(self, scope):  # DQN architechture  
        with tf.compat.v1.variable_scope(scope):
            state = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], "state")    #(1, 360)
            action = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim], "action")  #(1, 120)
            inputs = tf.concat([state, action], axis=-1)                                 #(1, 480) 
            layer1 = keras.layers.Dense(120, activation='relu', use_bias=False,
                                           kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
            layer2 = keras.layers.Dense(32, activation='relu', use_bias=False,
                                           kernel_regularizer=keras.regularizers.l2(0.01))(layer1)
            q_value = tf.compat.v1.layers.Dense(1, activation=None, use_bias=False)(layer2)
            return state, action, q_value

    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.q_value, self.loss, self.optimizer], feed_dict={self.state: state, self.action: action, self.predicted_q_value: predicted_q_value})

    def predict(self, state, action):
        return self.sess.run(self.q_value, feed_dict={self.state: state, self.action: action})

    def predict_target(self, state, action):
        return self.sess.run(self.target_q_value, feed_dict={self.target_state: state, self.target_action: action})

    def action_gradients(self, state, action):
        return self.sess.run(self.a_gradient, feed_dict={self.state: state, self.action: action})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)


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


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.compat.v1.summary.scalar("reward", episode_reward)
    critic_loss = tf.Variable(0.)
    tf.compat.v1.summary.scalar("critic_loss", critic_loss)

    summary_vars = [episode_reward, critic_loss]
    summary_ops = tf.compat.v1.summary.merge_all()
    return summary_ops, summary_vars


def learn_from_batch(replay_buffer, actor, critic, batch_size, s_dim, a_dim):

    #print('---------- Learn from Batch ----------')
    samples = replay_buffer.sample_batch(batch_size)
    state_batch = np.array([_[0] for _ in samples])        # (32, 360) 
    action_batch = np.array([_[1] for _ in samples])       # (32, 120)
    reward_batch = np.array([_[2] for _ in samples])       # (32,)
    next_state_batch = np.array([_[3] for _ in samples])
    done_batch = np.array([_[4] for _ in samples])

    # The dimension of action_weight_batch is (batch_size, action_item_num, embedding), (32, 4, 30)
    next_action_batch = actor.predict_target(state_batch) 
    
    # Q(s',a')
    target_q_batch = critic.predict_target(next_state_batch.reshape((-1, s_dim)), next_action_batch.reshape((-1, a_dim)))  # (32, 1)

    # y_batch is Q(s,a) calcualed from Bellman's equation
    y_batch = np.add(target_q_batch * critic.gamma, (reward_batch * done_batch)[:,np.newaxis])  # (32, 1)

    # train Critic online network (learn an action-value function Q(state, action)), q_value seems to be the functional approximation of y_batch
    q_value, critic_loss, _ = critic.train(state_batch, action_batch, y_batch)

    # train actor
    action_batch_for_gradients = actor.predict(state_batch)

    # action_gradient
    action_gradient_batch = critic.action_gradients(state_batch, action_batch_for_gradients.reshape((-1, a_dim)))   # action_gradient_batch, list, size = 1
    actor.train(state_batch, action_gradient_batch[0])

    # update target networks
    actor.update_target_network()
    critic.update_target_network()
    # return q value and critic loss from batch
    return critic_loss


def train(sess, env, actor, critic, s_dim, a_dim, global_step_tensor, args):
    # set up summary operators
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(args['summary_dir'], sess.graph)

    # initialize target network weights
    actor.hard_update_target_network()
    critic.hard_update_target_network()

    # initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']))

    # create saver object
    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    # restore the latest checkpoint of the model
    if bool(args['restore']) == True:
        saver.restore(sess, tf.train.latest_checkpoint('./model'))
    
    for i in range(int(args['max_episodes'])):
        state = env.reset()   # take one random sample from simulator
        ep_reward = 0.0
        ep_critic_loss = 0.0

        for j in range(args['max_episodes_len']):
            start_time = time.time()  

            action = actor.predict(np.reshape(state, [1, s_dim]))
            ## noise    
            
            next_state, reward, done, info = env.step(action[0])
            ep_reward += reward
           
            replay_buffer.add(state.reshape((s_dim,)),
                              action.reshpe((a_dim,)),
                              reward,
                              next_state.reshape((s_dim,)),  #(1, 8) to (8, )
                              done)
            
            if replay_buffer.size() > int(args['batch_size']):
                critic_loss = learn_from_batch(replay_buffer, actor, critic, int(args['batch_size']), s_dim, a_dim)
                ep_critic_loss += critic_loss

            # move to next state 
            state = next_state
            ## global step += 1
            current_step = tf.compat.v1.train.global_step(sess, global_step_tensor) - 1 
            #print('current_step is %d' %current_step) 

            summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward, summary_vars[1]: ep_critic_loss})
            writer.add_summary(summary_str, i)

            end_time = time.time()

            if done:
                break
            #print(f"--- Total Training Time : {(end_time - start_time):.3f} seconds ---")
                    
        logger.info(f"========== Episode {current_step // 100}, Total {j+1} Rounds : Reward {ep_reward:.3f}, Loss {ep_critic_loss:.3f} =========")
        # save the model to the checkpoint file
        path = saver.save(sess, './model/drl-model', global_step=current_step) 
    writer.close()


def main(args):
    # init memory data
    # data = load_data()
    
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.90)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
    ## specify particular gpu to use     
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

        # gym environment
        env = gym.make('LunarLanderContinuous-v2')
        # state total dimension  
        s_dim = 8
        # action toal dimension
        a_dim = 2
        # initialize global_step variable with 0
        global_step_tensor = tf.Variable(0, name="global_step", trainable=False)

        actor = Actor(sess, s_dim, a_dim, int(args['batch_size']), float(args['tau']),
                      float(args['actor_lr']), global_step_tensor = global_step_tensor )

        critic = Critic(sess, s_dim, a_dim, actor.get_num_trainable_vars(),
                        float(args['gamma']), float(args['tau']), float(args['critic_lr']))

        #exploration = StochasticPolicy(a_dim, int(args['action_item_num']))

        train(sess, env, actor, critic, s_dim, a_dim, global_step_tensor, args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--actor_lr", help="actor network learning rate", default=0.001)
    parser.add_argument("--critic_lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.75)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)  # default = 64

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=500)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=1000)    # defult = 100
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')
    parser.add_argument("--restore", help="restore from previous trained model", default = False)

    args_ = vars(parser.parse_args())
    logger.info(pp.pformat(args_))

    main(args_)





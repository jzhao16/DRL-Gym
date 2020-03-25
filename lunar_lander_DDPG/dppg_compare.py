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
                    tf.gradients(self.action, self.network_params, -self.a_gradient)                           # chain_rule
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
            hidden1 = keras.layers.Dense(400, activation='relu')(state)
            batchNorm1 = tf.keras.layers.BatchNormalization(axis=-1)(hidden1)
            hidden2 = keras.layers.Dense(200, activation='relu')(batchNorm1)
            batchNorm2 = tf.keras.layers.BatchNormalization(axis=-1)(hidden2)
            output = keras.layers.Dense(self.a_dim, activation='tanh')(batchNorm2)
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
            layer1 = keras.layers.Dense(400, activation='relu')(state)
            batchNorm1 = tf.keras.layers.BatchNormalization(axis=-1)(layer1)
            layer2 = keras.layers.Dense(200, activation='relu')(batchNorm1)
            layer3 = keras.layers.Dense(200, activation='relu')(action)
            concat = tf.concat([layer2, layer3], 1)
            q_value = tf.compat.v1.layers.Dense(1, activation=None)(concat)
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

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class OUNoise:
    def __init__(self, mu, theta=.2, sigma=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):

    sess.run(tf.compat.v1.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size, 0)

    max_episodes = ep
    max_steps = 3000
    score_list = []

    for i in range(max_episodes):

        state = env.reset()
        score = 0

        for j in range(max_steps):

            # env.render()

            action = actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()
            print(f"action : {action.shape}")
            next_state, reward, done, info = env.step(action[0])
            print(f"next_state : {next_state.shape}")
            replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                              done, np.reshape(next_state, (actor.s_dim,)))

            # updating the network in batch
            if replay_buffer.size() < min_batch:
                continue

            states, actions, rewards, dones, next_states = replay_buffer.sample_batch(min_batch)
            target_q = critic.predict_target(next_states, actor.predict_target(next_states))

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1-dones[k]))

            # Update the critic given the targets
            predicted_q_value, _, _ = critic.train(states, actions, np.reshape(y, (min_batch, 1)))

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            state = next_state
            score += reward

            if done:
                print('Reward: {} | Episode: {}/{}'.format(int(score), i, max_episodes))
                break

        score_list.append(score)

        avg = np.mean(score_list[-100:])
        print("Average of last 100 episodes: {0:.2f} \n".format(avg))

        if avg > 200:
            print('Task Completed')
            break

    return score_list


if __name__ == '__main__':

    with tf.compat.v1.Session() as sess:

        env = gym.make('LunarLanderContinuous-v2')

        env.seed(0)
        np.random.seed(0)
        tf.compat.v1.set_random_seed(0)

        ep = 2000
        tau = 0.001
        gamma = 0.99
        min_batch = 64
        actor_lr = 0.00005
        critic_lr = 0.0005
        buffer_size = 1000000

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = Actor(sess, state_dim, action_dim, min_batch, tau, actor_lr)
        critic = Critic(sess, state_dim, action_dim, actor.get_num_trainable_vars(), gamma, tau, critic_lr)
        scores = train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep)



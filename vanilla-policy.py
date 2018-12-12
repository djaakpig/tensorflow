import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
gamma = 0.99

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0,r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,biases_initializer=None,activation_fn=tf.nn.softmax)
        self.chosen_action = tf.nn.softmax(self.output,1)

        self.reward_h = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_h = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_h
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_h)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for i,v in enumerate(tvars):
            p = tf.placeholder(tf.float32,name=str(i)+'_holder')
            self.gradient_holders.append(p)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph()
myagent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8)

total_episodes = 5000
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

with tf.Session() as ss:
    ss.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradbuf = ss.run(tf.trainable_variables())
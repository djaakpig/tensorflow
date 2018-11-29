import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class context_bandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0, 0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getbandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def getreward(self,action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

class agent():
    def __init__(self, lr, s_size, a_size):
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_oh = slim.one_hot_encoding(self. state_in, s_size)
        output = slim.fully_connected(state_in_oh,
                                      a_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        self.reward_h = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_h = tf.placeholder(shape=[1], dtype=tf.int32)
        self.ressponsible_weights = tf.slice(self.output, self.action_h, [1])
        self.loss = -(tf.log(self.ressponsible_weights) * self.reward_h)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

tf.reset_default_graph()

the_bandit = context_bandit()
the_agent = agent(lr=0.001, s_size=the_bandit.num_bandits, a_size=the_bandit.num_actions)
weights = tf.trainable_variables()[0]

total_episodes = 10000
total_reward = np.zeros([the_bandit.num_bandits, the_bandit.num_actions])
e = 0.1

init = tf.global_variables_initializer()

with tf.Session() as ss:
    ss.run(init)

    i = 0
    while i < total_episodes:
        s = the_bandit.getbandit()

        if np.random.rand(1) < e:
            action = np.random.randint(the_bandit.num_actions)
        else:
            action = ss.run(the_agent.chosen_action, feed_dict={the_agent.state_in:[s]})

        reward = the_bandit.getreward(action)

        feed_dict = {the_agent.reward_h:[reward],
                     the_agent.action_h:[action],
                     the_agent.state_in:[s]}
        _, ww = ss.run([the_agent.update, weights], feed_dict=feed_dict)

        total_reward[s, action] += reward

        if i%500 == 0:
            print("Mean reward for each of the " + str(the_bandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis=1)))

        i += 1

for a in range(the_bandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a]) + 1) + " for bandit " + str(a + 1) + " is the most promising...")

    if np.argmax(ww[a]) == np.argmin(the_bandit.bandits[a]):
          print("...and it was right!")
    else:
          print("...and it was wrong!")

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

    gradbuf = ss.run(tf.trainable_variables())
    for ix, grad in enumerate(gradbuf):
        gradbuf[ix] = grad * 0

    i = 0
    total_reward = []
    total_length = []

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep): # 0 ~ max_ep-1 의 범위를 생성.
            # 네트워크 출력에서 확률적으로 액션을 구한다.
            a_dist = ss.run(myagent.output,feed_dict={myagent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0]) # p = a_dist 각 요소의 확률을 지정.
            a = np.argmax(a_dist==a) # [a_dist==a] a 와 같은 요소는 true, 나머지는 false 인 배열 생성.

            # 주어진 밴딧에 대해 액션을 취한 보상을 얻는다.
            s1,r,d,_ = env.step(a)
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            if d == True:
                # 네트워크를 업데이트한다.
                ep_history_arr = np.array(ep_history)
                ep_history_arr[:,2] = discount_rewards(ep_history_arr[:,2])
                feed_dict = {
                    myagent.reward_h:ep_history_arr[:,2],
                    myagent.action_h:ep_history_arr[:,1],
                    myagent.state_in:np.vstack(ep_history_arr[:,0])}
                grads = ss.run(myagent.gradients, feed_dict=feed_dict)
                for ix,grad in enumerate(grads):
                    gradbuf[ix] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myagent.gradient_holders, gradbuf))
                    _ = ss.run(myagent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradbuf):
                        gradbuf[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

        # 보상의 총계 업데이트
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1

import tensorflow as tf
import numpy as np

bandit_arms = [0.2,-2,-0.2,0]
num_arms = len(bandit_arms)

def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

reward_h = tf.placeholder(shape=[1],dtype=tf.float32)
action_h = tf.placeholder(shape=[1],dtype=tf.int32)

response_o = tf.slice(output,action_h,[1])
loss = -(tf.log(response_o)*reward_h)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)

total_episodes = 1000
total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer()

with tf.Session() as ss:
    ss.run(init)
    i = 0
    while i < total_episodes:
        actions = ss.run(output)
        a = np.random.choice(actions,p=actions)
        action = np.argmax(actions == a)

        reward = pullBandit(bandit_arms[action])

        _, resp, ww = ss.run([update,response_o,weights],feed_dict={reward_h:[reward],action_h:[action]})

        total_reward[action] += reward
        if i%50 == 0:
            print("Running reward for the " + str(num_arms) + " arms of the bandit: " +
                    str(total_reward))
        i += 1
print("\nThe agent thinks arms " + str(np.argmax(ww)+1) + " is the most promising...")
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print("and it was right!")
else:
    print("and it was wrong!")


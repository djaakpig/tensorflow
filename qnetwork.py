import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

# 액션을 선택하는 데 사용되는 네트워크의 피드포워드 부분.
inputs = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
outQ = tf.matmul(inputs, W)
predict = tf.argmax(outQ,1)

# 타겟 Q 값과 예측 Q 값의 차의 제곱합을 구함으로써 비용을 얻는다.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - outQ))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss) # mine : 그래프 내 학습가능한 변수들을 학습시킨다.

init = tf.global_variables_initializer()

# 학습 매개변수를 설정한다.
y = 0.99
e = 0.1
num_episodes = 2000

# 보상의 총계와 에프소드 별 단계 수를 담을 리스트.
rewards = []

with tf.Session() as ss:
    ss.run(init)
    for i in range(num_episodes):
        # 환경을 리셋하고 첫 번째 새로운 관찰을 얻는다.
        s = env.reset()
        sum_reward = 0
        d = False
        j = 0
        # Q 네트워크
        while j < 99:
            j += 1
            # Q 네트워크에서 (e의 확률로 랜덤한 액션과 함께) 그리디하게 액션을 선택.
            a,Q = ss.run([predict,outQ],
                         feed_dict={inputs:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # 환경으로 부터 새로운 상태와 보상을 얻는다.
            s1,r,d,_ = env.step(a[0])
            # 새로운 상태를 네트워크에 피드해줌으로써 Q' 값을 구한다.
            Q1 = ss.run(outQ,
                        feed_dict={inputs:np.identity(16)[s1:s1+1]})
            # maxQ' 값을 구하고 선택된 액션에 대한 타깃 값을 설정한다.
            maxQ1 = np.max(Q1)
            targetQ = Q
            targetQ[0,a[0]] = r + y*maxQ1
            # 타깃 및 예측 Q 값을 이용해 네트워크를 학습시킨다.
            _, W1 = ss.run([updateModel,W],
                           feed_dict={inputs:np.identity(16)[s:s+1],nextQ:targetQ})
            sum_reward += r
            s = s1
            if d == True:
                # 모델을 학습해 나감에 따라 랜덤 액션의 가능성을 줄여간다.
                e = 1./((i/50) + 10)
                break
        # jList.append(j)
        rewards.append(sum_reward)

print("Percent of succesful episodes: " + str(sum(rewards)/num_episodes))
plt.plot(rewards)

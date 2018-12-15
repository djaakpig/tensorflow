import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# 테이블을 0ㅇ로 초기화
Q = np.zeros([env.observation_space.n,env.action_space.n])
# 학습할 매개변수를 설정한다.
lr = .85
y = .99
num_episodes = 2000
# 보상의 총계를 담을 리스트.
rewards = []
for i in range(num_episodes):
    # 환경을 리셋하고 첫 번째 새로운 관찰자를 얻는다.
    s = env.reset()
    sum_reward = 0
    d = False
    j = 0
    # Q 테이블 학습 알고리즘
    while j < 99:
        j += 1
        # Q 테이블로 부터 (노이즈와 함께) 그리디하게 액션을 선택.
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n) * (1/(i + 1)))
        # 환경으로 부터 새로운 상태와 보상을 얻는다.
        s1,r,d,_ = env.step(a)
        # 새로운 지식을 통해 Q 테이블을 업데이트한다.
        Q[s,a] = Q[s,a] + lr * (r + y * np.max(Q[s1,:]) - Q[s,a])
        sum_reward += r
        s = s1
        if d == True:
            break
rewards.append(sum_reward)
print("Score over time: " + str(sum(rewards)/num_episodes))
print("Final Q-Table Values ")
print(Q)

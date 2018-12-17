import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import os

from gridworld import GameEnv

env = GameEnv(partial=False, size=5)

class QNetwork():
    def __init__(self, hSize):
        # 네트워크는 게임으로 부터 하나의 프레임을 받아 이를 배열로 만든다.(flattening)
        # 그 다음 배열의 크기를 재조절하고 4개의 합성곱 계층을 거쳐 처리한다.
        self.scalarInput = tf.placeholder(shape=[None,21168], dtype=tf.float32)
        self.imageInput = tf.reshape(self.scalarInput, shape=[-1,84,84,3])
        with slim.arg_scope([slim.conv2d], padding='VALID', biases_initializer=None):
            self.conv1 = slim.conv2d(inputs=self.imageInput,
                                        num_outputs=32,
                                        kernel_size=[8,8],
                                        stride=[4,4])
            self.conv2 = slim.conv2d(inputs=self.conv1,
                                        num_outputs=64,
                                        kernel_size=[4,4],
                                        stride=[2,2])
            self.conv3 = slim.conv2d(inputs=self.conv2,
                                        num_outputs=64,
                                        kernel_size=[3,3],
                                        stride=[1,1])
            self.conv4 = slim.conv2d(inputs=self.conv3,
                                        num_outputs=hSize,
                                        kernel_size=[7,7],
                                        stride=[1,1])

        # 마지막 합성곱 계층에서 출력값을 취한 후 이를 어드밴티지 스트림과 가치 스트림으로 분리.
        self.streamAC,self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        self.aw = tf.Variable(tf.random_normal([hSize//2,env.actions]))
        self.vw = tf.Variable(tf.random_normal([hSize//2,1]))
        self.advantage = tf.matmul(self.streamA, self.aw)
        self.value = tf.matmul(self.streamV, self.vw)

        # 최종 Q 값을 얻기 위해 어드밴티지 스트림과 가치 스트림을 조합.
        advantageMean = tf.reduce_mean(self.advantage, axis=1, keepdims=True)
        self.outQ = self.value + tf.subtract(self.advantage, advantageMean)
        self.predict = tf.argmax(self.outQ, 1)

        # 타깃 Q 값과 예측 Q 값의 차의 제곱합을 구함으로써 비용을 얻는다.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.onehotActions = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        self.Q = tf.reduce_mean(tf.multiply(self.outQ, self.onehotActions), axis=1)
        self.error = tf.square(self.targetQ-self.Q)
        self.loss = tf.reduce_mean(self.error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class ExperienceBuffer():
    def __init__(self, size=50000):
        self.buffer = []
        self.size = size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.size]=[]
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size,5])

def processState(states):
    return np.reshape(states, [21168])

def updateTargetGraph(tfVars, tau):
    halfVars = len(tfVars)//2
    opHolders = []
    for i,v in enumerate(tfVars[0:halfVars]):
        opHolder = (v.value()*tau) + ((1-tau)*tfVars[i+halfVars].value())
        opHolders.append(tfVars[i+halfVars].assign(opHolder))
    return opHolders

def updateTarget(opHolders, ss):
    for op in opHolders:
        ss.run(op)

batchSize = 32 # 각 학습 단계에서 사용할 경험 배치의 수.
updateFreq = 4 # 학습 단계 업데이트 빈도
y = .99 # 타깃 Q 값에 대한 할인 계수.
startE = 1 # 랜덤한 액션을 시작할 가능성.
endE = .1 # 랜덤한 액션을 끝낼 가능성.
annelingSteps = 10 # startE 에서 endE 로 줄어드는 데 필요한 학습 단계 수.
numEpisodes = 100 # 네트워크를 학습시키기 위한 게임 환경 에피소드 수.
preTrainSteps = 100 # 학습 시작 전 랜덤 액션의 단계 수`
maxEpisodes = 20 # 허용되는 최대 에피소드 길이.
loadModel = False # 저장된 모델을 로드할지 여부.
path = './dqn' # 모델을 저장할 위치.
hSize = 512 # 어드밴티지/가치 스트림으로 분리되기 전 마지막 합성곱 계층의 크기.
tau = .001 # 타깃 네트워크를 제1 네트워크로 업데이트하는 비율.

tf.reset_default_graph()
mainQN = QNetwork(hSize)
targetQN = QNetwork(hSize)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainableVars = tf.trainable_variables()
targetOps = updateTargetGraph(trainableVars, tau)
historyBuffer = ExperienceBuffer()

# 랜덤 액션이 감소하는 비율을 설정.
e = startE
stepDrop = (startE-endE)/annelingSteps

# 보상의 총계와 에피소드 별 단계 수를 담을 리스트를 생성.
rewardSteps = []
rewards = []
totalSteps = 0

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as ss:
    ss.run(init)
    if loadModel == True:
        print('loading model...')
        cp = tf.train.get_checkpoint_state(path)
        saver.restore(ss, cp.model_checkpoint_path)
    # 타깃 네트워크가 제1 네트워크와 동일하도록 설정.
    updateTarget(targetOps, ss)
    for i in range(numEpisodes):
        episodeBuffer = ExperienceBuffer()
        # 환경을 리셋하고 첫 번째 새로운 관찰을 얻는다.
        s = env.reset()
        s = processState(s)
        d = False
        sumRewards = 0
        j = 0
        # Q 네트워크
        # 에이전트가 블록에 도달하기까지 최대 50호 시도하고 종료.
        while j < maxEpisodes:
            j += 1
            # Q 네트워크에서 (e의 확률로 랜덤한 액션과 함께) 그리디하게 액션을 선택.
            if np.random.randn(1) < e or totalSteps < preTrainSteps:
                a = np.random.randint(0,4)
            else:
                feed_dict = {mainQN.scalarInput:[s]}
                predict = ss.run(mainQN.predict, feed_dict)
                a = predict[0]
            s1,r,d = env.step(a)
            s1 = processState(s1)
            totalSteps += 1
            # 에피소드 버퍼에 경험을 저장.
            experience = np.reshape(np.array([s,a,r,s1,d]), [1,5])
            episodeBuffer.add(experience)

            if totalSteps > preTrainSteps:
                if e > endE:
                    e -= stepDrop

                if totalSteps % updateFreq == 0:
                    # 경험에서 랜덤하게 배치 하나를 샘플링.
                    trainBatches = historyBuffer.sample(batchSize)
                    # 타깃 Q 값에 대해 더블 DQN 업데이트를 수행.
                    input = np.vstack(trainBatches[:,3])
                    Q1 = ss.run(mainQN.predict, feed_dict={mainQN.scalarInput:input})
                    Q2 = ss.run(targetQN.outQ, feed_dict={targetQN.scalarInput:input})
                    endMultiplier = -(trainBatches[:,4]-1)
                    doubleQ = Q2[range(batchSize),Q1]
                    targetQ = trainBatches[:,2] + (y*doubleQ*endMultiplier)
                    # 타깃 값을 이용해 네트워크를 업데이트.
                    feed_dict = {
                        mainQN.scalarInput:np.vstack(trainBatches[:,0]),
                        mainQN.targetQ:targetQ,
                        mainQN.actions:trainBatches[:,1]
                    }
                    _ = ss.run(mainQN.updateModel, feed_dict=feed_dict)
                    # 타깃 네트워크가 제1 네트워크와 동일하도록 설정.
                    updateTarget(targetOps, ss)
            sumRewards += r
            s = s1

            if d == True:
                break

        historyBuffer.add(episodeBuffer.buffer)
        rewardSteps.append(j)
        rewards.append(sumRewards)

        # 정기적으로 모델 저장.
        if i % 1000 == 0:
            saver.save(ss, path+'/model-'+str(i)+'.cp')
            print('saved model')
        if len(rewards) % 10 == 0:
            print(totalSteps, np.mean(rewards[-10:]), e)
    saver.save(ss, path+'/model-'+str(i)+'.cp')

print('Percent of successful episodes: '+str(sum(rewards)/numEpisodes))
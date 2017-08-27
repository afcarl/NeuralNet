from links import *
from model import Sequential
from dqn import DQN
import numpy as np
from numpy.random import *
import gym
from copy import deepcopy

env = gym.make("CartPole-v0")
obs = env.reset()

model = Sequential()
model.add(Linear(4,400,activation="relu",initialization="HeNormal"))
#model.add(Linear(400,400,activation="relu",initialization="HeNormal"))
#model.add(Linear(100,100,activation="relu",initialization="HeNormal"))
model.add(Linear(400,2,initialization="zeros"))
model.compile(optimizer="MomentumSGD")
target_model = deepcopy(model)
Memory = DQN()
initial_exploration = 100
replay_size = 32

epsilon = 0.3
gamma = 0.95
time = 0
episode = 0
last_obs = deepcopy(obs)

#ReplayMemory = [None for i in range(10**5)]
#m_size = 0
step = 0
while True:
    #env.render()
    step += 1
    time += 1
    #env.render()
    Q = model(obs,save=False)
    #print Q[0]
    if epsilon > rand() or step < 100:
        action = randint(0,2)
    else:
        action = np.argmax(Q[0])
    epsilon -= 2e-4
    if epsilon < 0.:
        epsilon = 0.
    obs, reward, done, _ = env.step(action)
    reward = 0.
    if done:
        reward = -1.
        episode += 1

    Memory.add(last_obs,action,reward,obs,done)

    if done:
        obs = env.reset()
    last_obs = deepcopy(obs)
    if done and episode % 100 == 0:
        print 'episode:',episode,'step:',step,'eps:',epsilon,'ave:',time/100.,'Q:',Q[0]
        time = 0.

    #t = deepcopy(Q)
    if step < 100:
        continue
    sample = [Memory.ReplayMemory[(Memory.count-1)%10**6]]#sample(16)
    #sample = []
    #for i in range(10):
    #    sample.append(Memory.ReplayMemory[np.random.randint(0,min(10**6,Memory.count-1))])
    #sample = Memory.sample(32)
    #print len(sample)
    loss = 0
    for s in sample:
        Q = model(s[0])
        t = deepcopy(Q)
        if s[4]:
            t[0][s[1]] = s[2]
            loss = Q - t
            model.update(loss)
        else:
            next_Q = target_model(s[3],save=False)
            t[0][s[1]] = s[2] + gamma * np.max(next_Q[0])
            loss = Q - t
            model.update(loss)
    #model.update(loss)

    if step % 10 == 0:
        target_model = deepcopy(model)

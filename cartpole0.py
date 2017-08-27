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
model.add(Linear(4,500,activation="relu",initialization="HeNormal"))
#model.add(Linear(200,200,activation="relu",initialization="HeNormal"))
#model.add(Linear(100,100,activation="relu",initialization="HeNormal"))
model.add(Linear(500,2,initialization="zeros"))
model.compile(optimizer="Adam")
Memory = DQN()

epsilon = 0.
gamma = 0.95
time = 0
episode = 0
last_obs = deepcopy(obs)

#ReplayMemory = [None for i in range(10**5)]
#m_size = 0

while True:
    time += 1
    #env.render()
    Q = model(obs)
    #print Q[0]
    if epsilon > rand():
        action = randint(0,2)
    else:
        action = np.argmax(Q[0])
    epsilon -= 1e-6
    if epsilon < 0.:
        epsilon = 0.
    obs, reward, done, _ = env.step(action)
    t = deepcopy(Q)
    if done:
        reward = -1.
        t[0][action] = reward
        loss = Q - t
        model.update(loss)
        obs = env.reset()
        episode += 1
        if episode % 100 == 0:
            print 'episode:',episode,'eps:',epsilon,'ave:',time/100.,'Q:',Q[0]
            time = 0.
    else:
        reward = 0.
        next_Q = model(obs,save=False)
        t[0][action] = reward + gamma * np.max(next_Q[0])
        loss = Q - t
        model.update(loss)
    last_obs = deepcopy(obs)

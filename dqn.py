import numpy as np
import random

class DQN:
    def __init__(self,size=10**6):
        self.ReplayMemory = [None for i in range(size)]
        self.size = size
        self.count = 0
    def add(self,s,a,r,s_dash,done):
        self.ReplayMemory[self.count%self.size] = [s,a,r,s_dash,done]
        self.count += 1
    def sample(self,num):
        return random.sample(self.ReplayMemory[0:min(self.size,self.count)],num)

from links import *
from model import Sequential
import numpy as np
from numpy.random import *
import chainer
from tqdm import tqdm

model = Sequential()
model.add(Linear(784,500,activation="relu",initialization="HeNormal"))
model.add(Linear(500,500,activation="relu",initialization="HeNormal"))
model.add(Linear(500,10,activation="softmax"))
model.compile(optimizer="Adam")

train, test = chainer.datasets.get_mnist()
train_data, train_label = train._datasets
test_data, test_label = test._datasets
#print train_label[0:100]

count = 0
count2 = 0
loss = 0
for i in tqdm(range(6000000)):
    #if train_label[i%60000]>1:
    #    continue
    #count2 += 1
    #inp = randint(0,2,(1,2))
    inp = np.zeros((1,784))
    inp[0] = train_data[i%60000]
    y = model(inp)
    t = np.zeros((1,10))
    #t[0][0] = train_label[i%60000]
    t[0][train_label[i%60000]] = 1.
    #t = np.zeros((1,1))
    #if int(inp[0][0]) ^ int(inp[0][1]):
    #    t[0][0] = 1.
    #inp = inp.astype(np.float32)
    loss += y - t
    if i%100:
        model.update(loss/100.)
        loss = 0
    #print loss
    #if y[0][0] > 0.5 and t[0][0] > 0.5 or y[0][0] < 0.5 and t[0][0] < 0.5:
    #        count += 1
    #print np.argmax(y[0])
    #print y[0]
    if np.argmax(y[0]) == train_label[i%60000]:
        count += 1
    if i%60000 == 0 and i != 0:
        print count/60000.
        count = 0

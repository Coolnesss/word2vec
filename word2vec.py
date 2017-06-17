from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import tensorflow as tf

def custom_objective(y_true, y_pred):
    return -(y_true * tf.log(y_pred))

words = "this is a great white shark".split()

N = len(words)
v_size = len(set(words))

model = Sequential()

model.add(Dense(300, input_shape=(v_size, )))
model.add(Dense(v_size, activation="softmax"))

model.compile(loss=custom_objective, optimizer='adadelta')

w_size = 1

X = np.zeros((N, v_size))
Y = np.zeros((N, v_size))
#X = []

vocab = dict()
index2word = dict()

for idx, i in enumerate(set(words)):
    vocab[i] = idx
    index2word[idx] = i

for idx, i in enumerate(words):
    X[idx, vocab[i]] = 1

    for j in range(1, (w_size + 1)):
        if idx - j >= 0:
            asd = vocab[words[idx - j]]
            Y[idx, asd] = 1
        if idx + j < len(words):
            asd = vocab[words[idx + j]]
            Y[idx, asd] = 1

model.fit(x=X, y=Y, batch_size=1)

print model.predict(X[:1])
print index2word[np.argmax(model.predict(X[2:3]))]

from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import zipfile
from keras import metrics
import keras.backend as K


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
filename = "data/text8.zip"
words = read_data(filename)
print('Data size', len(words))

def custom_objective(y_true, y_pred):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.cast(tf.equal(y_true, zero), tf.float32)
    return -(tf.log(y_true * y_pred) + tf.log(where * y_pred))

#words = "this is a great white shark".split()

N = min(1000, len(words))
words  = words[:N]
v_size = len(set(words))
print N * v_size

model = Sequential()

model.add(Dense(300, input_shape=(v_size, )))
model.add(Dense(v_size, activation="softmax"))

model.compile(loss=custom_objective, optimizer='adadelta')

w_size = 3

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

model.fit(x=X, y=Y, batch_size=64, epochs=50)

n_test = 100
y_pred = model.predict(X[:n_test])
y_true = Y[:n_test]

score = 0
for i, row in enumerate(y_pred):
    ind = np.argsort(-row)[:w_size * 2]
    print ind == np.where(y_true[i] == 1)
    score += np.sum(ind == np.where(y_true[i] == 1)) / (w_size * 2.0)
    print score
print score / float(y_pred.shape[0])
print words[:5]
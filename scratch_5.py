from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
from scipy.io import loadmat
nums = range(100, 235)
dat = list()
y = []
for num in nums:
    try:
        dat.append(loadmat('/home/daniel/Downloads/QRS_DAT/mat_dat/' + str(num) + 'm.mat')['val'])
        with open('/home/daniel/Downloads/QRS_DAT/physionet.org/files/mitdb/1.0.0/' + str(num) + '.hea') as f:
            if 'Digoxin' in f.read():
                y.append(1)
            else:
                y.append(0)
    except:
        continue
print(dat)
y = np.asarray(y)
trainY= y[:int(len(y)/2)]
evalY = y[int(len(y)/2):]
X = [np.asarray([int(v) for v in d[0]]) for d in dat]
print(X)
trainX = X[:int(len(y)/2)]
evalX = X[int(len(y)/2):]
print(len(X))
print(y)
# print(X)

inp = layers.Input(shape=(len(X[0]), 1))
x = layers.Conv1D(64, 3, activation='relu')(inp)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(128, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inp, output)
model.summary()

model.compile(optimizer='SGD', loss='hinge', metrics=['acc'])

model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=0)
_, accuracy = model.evaluate(evalX, evalY, batch_size=1, verbose=0)

print(accuracy)


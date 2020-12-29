import numpy as np
from sklearn import datasets

import deepops as dp
from deepops.model import Model


class StupidLittleModel(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = dp.layers.Dense(4, 16, activation="relu", name="dense1")
        self.dense2 = dp.layers.Dense(16, 16, activation="relu", name="dense1")
        self.dense3 = dp.layers.Dense(16, 1, activation="sigmoid", name="dense2")

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


train_X, train_y = datasets.load_iris(return_X_y=True)
X, y = np.asarray(train_X[:100]), np.asarray(train_y[:100])
yi = np.argwhere(y <= 1)
y = np.reshape(y[yi], (-1))
X = np.reshape(X[yi], (y.shape[0], -1))
X = (X - X.min()) / (X.max() - X.min())
X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)


def loss(y, ypred):
    # MSE
    _loss = [(yb - ypb) * (yb - ypb) for yb, ypb in zip(y, ypred)]
    return sum(_loss) * dp.Tensor(1 / len(yb))


sequential = StupidLittleModel()
batch_size = 20


for steps in range(1000):
    ri = np.random.permutation(X.shape[0])[:batch_size]
    xb = X[ri]
    yb = y[ri]
    xb = [list(map(dp.Tensor, x)) for x in xb]
    # forward
    y_pred_b = list(map(sequential.forward, xb))
    yb = [dp.Tensor(y) for y in yb]
    total_loss = loss(yb, y_pred_b)
    # backward
    sequential.init_backward()
    total_loss.backward()
    # mini optimizer
    learning_rate = 0.1
    for p in sequential.parameters():
        p._data -= learning_rate * p.grad
    if steps % 1 == 0:
        print(f"step {steps} loss {total_loss.data}")

breakpoint()

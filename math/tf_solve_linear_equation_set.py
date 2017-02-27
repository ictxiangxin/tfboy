import tensorflow as tf
import numpy as np


class TFSolveLinearEquationSet:
    def __init__(self, rate: float, loss_threshold: float=0.0001, max_epochs: int=1000):
        self.__rate = rate
        self.__loss_threshold = tf.constant(loss_threshold)
        self.__max_epochs = max_epochs
        self.__session = tf.Session()
        self.__x = None
        self.__loss = 0

    def __del__(self):
        try:
            self.__session.close()
        finally:
            pass

    def loss(self):
        return self.__loss

    def solve(self, coefficients, b):
        a_data = np.array(coefficients)
        x_dim = a_data.shape[1]
        b_data = np.array(b).reshape([x_dim, 1])
        _a = tf.placeholder(tf.float32)
        _b = tf.placeholder(tf.float32)
        _x = tf.Variable(tf.zeros([x_dim, 1]))
        loss = tf.reduce_mean(tf.square(tf.matmul(_a, _x) - _b))
        optimizer = tf.train.GradientDescentOptimizer(self.__rate)
        model = optimizer.minimize(loss)
        self.__session.run(tf.global_variables_initializer())
        for epoch in range(self.__max_epochs):
            self.__session.run(model, {_a: a_data, _b: b_data})
            if epoch % 10 == 0:
                if self.__session.run(loss < self.__loss_threshold, {_a: a_data, _b: b_data}):
                    break
        self.__loss = self.__session.run(loss, {_a: a_data, _b: b_data})
        return self.__session.run(_x)

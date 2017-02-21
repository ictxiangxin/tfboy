import tensorflow as tf
import numpy as np


class TFSolveEquation:
    def __init__(self, rate: float, loss_threshold: float=0.0001, max_epochs: int=1000):
        self.__rate = rate
        self.__loss_threshold = tf.constant(loss_threshold)
        self.__max_epochs = max_epochs
        self.__session = tf.Session()
        self.__x = None

    def __del__(self):
        try:
            self.__session.close()
        finally:
            pass

    def solve(self, coefficients, b):
        coefficients_data = np.array(coefficients)
        b_data = np.array(b)
        _a = tf.placeholder(tf.float32)
        _b = tf.placeholder(tf.float32)
        _x = tf.Variable(tf.zeros([coefficients_data.shape[1], 1]))
        loss = tf.reduce_mean(tf.square(tf.matmul(_a, _x) - _b))
        optimizer = tf.train.GradientDescentOptimizer(self.__rate)
        model = optimizer.minimize(loss)
        self.__session.run(tf.global_variables_initializer())
        for epoch in range(self.__max_epochs):
            self.__session.run(model, {_a: coefficients_data, _b: b_data})
            if epoch % 10 == 0:
                if self.__session.run(loss < self.__loss_threshold, {_a: coefficients_data, _b: b_data}):
                    break
        return self.__session.run(_x)

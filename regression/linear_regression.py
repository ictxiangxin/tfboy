import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, rate: float, loss_threshold: float=0.0001, max_epochs: int=1000):
        self.__rate = rate
        self.__loss_threshold = tf.constant(loss_threshold)
        self.__max_epochs = max_epochs
        self.__session = tf.Session()
        self.__w = None
        self.__b = None

    def __del__(self):
        try:
            self.__session.close()
        finally:
            pass


    def w(self):
        if self.__w is not None:
            return self.__session.run(self.__w)

    def b(self):
        if self.__b is not None:
            return self.__session.run(self.__b)

    def train(self, train_data, target):
        x_data = np.array(train_data)
        y_data = np.array(target)
        _x = tf.placeholder(tf.float32)
        _y = tf.placeholder(tf.float32)
        if len(x_data.shape) == 1:
            self.__w = tf.Variable(tf.zeros([1]))
        else:
            self.__w = tf.Variable(tf.zeros([x_data.shape[1], 1]))
        if len(y_data.shape) == 1:
            self.__b = tf.Variable(tf.zeros([1]))
        else:
            self.__b = tf.Variable(tf.zeros([y_data.shape, 1]))
        _f = tf.multiply(_x, self.__w) + self.__b
        loss = tf.reduce_mean(tf.square(_f - _y))
        optimizer = tf.train.GradientDescentOptimizer(self.__rate)
        model = optimizer.minimize(loss)
        self.__session.run(tf.global_variables_initializer())
        for epoch in range(self.__max_epochs):
            self.__session.run(model, {_x: x_data, _y: y_data})
            if epoch % 10 == 0:
                if self.__session.run(loss < self.__loss_threshold, {_x: x_data, _y: y_data}):
                    break

    def predict(self, predict_data):
        x_data = np.array(predict_data)
        _x = tf.placeholder(tf.float32)
        return self.__session.run(tf.multiply(_x, self.__w) + self.__b, {_x: x_data})

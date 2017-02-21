import numpy as np


class SolveEquation:
    def __init__(self, rate: float, loss_threshold: float=0.0001, max_epochs: int=1000):
        self.__rate = rate
        self.__loss_threshold = loss_threshold
        self.__max_epochs = max_epochs
        self.__x = None

    def solve(self, coefficients, b):
        _a = np.array(coefficients)
        _b = np.array(b).reshape([len(b), 1])
        _x = np.zeros([_a.shape[1], 1])
        for epoch in range(self.__max_epochs):
            grad_loss = np.matmul(np.transpose(_a), np.matmul(_a, _x) - _b)
            _x -= self.__rate * grad_loss
            if epoch % 10 == 0:
                loss = np.mean(np.square(np.subtract(np.matmul(_a, _x), _b)))
                print('loss = {:.8f}'.format(loss))
                if loss < self.__loss_threshold:
                    break
        return _x

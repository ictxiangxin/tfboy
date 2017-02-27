import numpy as np


class SolveEquation:
    def __init__(self, rate: float, loss_threshold: float=0.0001, max_epochs: int=1000, auto_rate: bool=True):
        self.__rate = rate
        self.__loss_threshold = loss_threshold
        self.__max_epochs = max_epochs
        self.__auto_rate = auto_rate
        self.__x = None
        self.__epochs = 0

    def rate(self):
        return self.__rate

    def epochs(self):
        return self.__epochs

    def solve(self, coefficients, b):
        _a = np.array(coefficients)
        x_dim = _a.shape[1]
        _b = np.array(b).reshape([x_dim, 1])
        _x = np.zeros([x_dim, 1])
        old_loss = np.mean(np.square(np.subtract(np.matmul(_a, _x), _b)))
        count = 0
        for epoch in range(self.__max_epochs):
            self.__epochs = epoch + 1
            if self.__auto_rate:
                loss = np.mean(np.square(np.subtract(np.matmul(_a, _x), _b)))
                if loss > old_loss:
                    self.__rate /= 2
                    count = 0
                else:
                    count += 1
                    if count > 10:
                        self.__rate *= 2
                        count = 0
                old_loss = loss
            grad_loss = np.matmul(np.transpose(_a), np.matmul(_a, _x) - _b)
            _x -= self.__rate * grad_loss
            if epoch % 10 == 0:
                loss = 0.5 * np.sum(np.square(np.subtract(np.matmul(_a, _x), _b)))
                print('loss = {:.8f}'.format(loss))
                if loss < self.__loss_threshold or loss is np.nan:
                    break
        return _x
